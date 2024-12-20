# ruff: noqa: F722
from typing import Optional
from jaxtyping import Shaped, Bool, Float
from torch import Tensor

from itertools import product

import triton
import triton.language as tl
import torch

__all__ = ["unfold"]


def unfold(
    x,
    block_size: tuple,
    stride: Optional[tuple] = None,
    mask: Optional[Bool[Tensor, "..."]] = None,
) -> Tensor:
    """Wrapper that dispatches complex and real tensors"""
    x_flat, shapes = prep_shapes(x, block_size, stride, mask)
    if torch.is_complex(x_flat):
        real = _unfold(x_flat, **shapes)
        imag = _unfold(x_flat, **shapes)
        y_flat = real + 1j * imag
    else:
        y_flat = _unfold(x_flat, **shapes)
    return y_flat.reshape(
        *shapes["batch_shape"],
        *shapes["nblocks"],
        *shapes["block_size"],
    )


def _unfold(
    x: Shaped[Tensor, "B ..."],
    block_size: tuple[int],
    stride: tuple[int, ...],
    ndim: int,
    im_size: tuple[int, ...],
    nblocks: tuple[int, ...],
    nbatch: int,
    # batch_shape: tuple[int, ...],
    mask: Bool[Tensor, "..."],
    **kwargs,
) -> Shaped[Tensor, "B ..."]:
    """Implementation of unfold"""
    if x.is_cuda and ndim in (1, 2, 3):
        with torch.cuda.device(x.device):
            # Allocate output
            y = torch.zeros(
                nbatch,
                *nblocks,
                *block_size,
                device=x.device,
                dtype=x.dtype,
            )
            grid = _get_grid(ndim, nbatch, nblocks)
            BLOCK_SIZE = tuple(
                triton.next_power_of_2(blk_size) for blk_size in block_size
            )
            UNFOLD[ndim][grid](
                x,
                y,
                nbatch,
                *nblocks,
                *block_size,
                *im_size,
                *stride,
                *BLOCK_SIZE,
            )
    else:
        y = _unfold_torch(x)
    return y


def _get_grid(ndim: int, nbatch, nblocks: tuple[int, ...]):
    if ndim == 1:
        grid = lambda meta: (  # noqa: E731
            nbatch * triton.cdiv(nblocks[0], meta["x_blocks_per_grid"]),
        )
    elif ndim == 2:
        grid = lambda meta: (  # noqa: E731
            nbatch * triton.cdiv(nblocks[0], meta["x_blocks_per_grid"]),
            triton.cdiv(nblocks[1], meta["y_blocks_per_grid"]),
        )
    elif ndim == 3:
        grid = lambda meta: (  # noqa: E731
            nbatch * triton.cdiv(nblocks[0], meta["x_blocks_per_grid"]),
            triton.cdiv(nblocks[1], meta["y_blocks_per_grid"]),
            triton.cdiv(nblocks[2], meta["z_blocks_per_grid"]),
        )
    else:
        raise ValueError(f"Invalid ndim = {ndim}")
    return grid


def get_configs():
    warps = [1, 2, 4, 8, 16]
    stages = [1, 2, 3, 4, 5]
    blocks_per_grid = [2**i for i in range(8)]
    return [
        triton.Config({"x_blocks_per_grid": bpg}, num_warps=warp, num_stages=stages)
        for (bpg, warp, stages) in product(blocks_per_grid, warps, stages)
    ]


@triton.autotune(
    configs=get_configs(),
    key=["x_block_dim", "x_size"],
)
@triton.jit
def _unfold1d(
    in_ptr,
    out_ptr,
    # Number of batches
    nbatch: int,
    # Number of blocks
    x_nblocks: int,
    # Size of each block
    x_block_dim: int,
    # Size of the input data
    x_size: int,
    # Stride of the blocks
    x_stride: int,
    # Size of the triton block (power of 2)
    X_BLOCK_SIZE: tl.constexpr,
    # Number of blocks per grid pid
    x_blocks_per_grid: int,
):
    pid_0 = tl.program_id(0)
    # Batch index, Block index
    NBx = pid_0 * x_blocks_per_grid
    N, Bx = NBx // x_nblocks, NBx % x_nblocks

    # Get block from input
    # x_load_range = tl.arange(0, X_BLOCK_SIZE)
    # x_load_mask = x_load_range < x_size
    # load_range = x_load_range
    # load_mask = x_load_mask
    # size = x_size
    in_offset = N * x_size + Bx * x_stride

    in_blk_ptr = tl.make_block_ptr(
        in_ptr + in_offset,
        shape=(nbatch, x_size),
        strides=(1, 1),
        offsets=(0, 0),
        block_shape=(1, X_BLOCK_SIZE),
        order=(0, 1),
    )
    blk_range = tl.arange(0, X_BLOCK_SIZE)
    blk_mask = blk_range < x_block_dim
    # out_offset = N * x_nblocks * x_block_dim + Bx * x_block_dim

    for i in range(x_blocks_per_grid):
        if Bx + i < x_nblocks:
            blk = tl.load(in_blk_ptr)
            # Save block to output
            out_offset = N * x_nblocks * x_block_dim + (Bx + i) * x_block_dim
            out_range = blk_range[None, :]
            out_mask = blk_mask[None, :]
            tl.store(out_ptr + out_offset + out_range, blk, out_mask)
        in_blk_ptr = tl.advance(in_blk_ptr, (0, x_stride))


UNFOLD = {1: _unfold1d}


@torch.compile
def _unfold_torch(
    x: Float[Tensor, "B ..."],
    block_size: tuple[int, ...],
    stride: tuple[int, ...],
    ndim: int,
    im_size: tuple[int, ...],
    nblocks: tuple[int, ...],
    nbatch: int,
    mask: Bool[Tensor, "..."],
) -> Float[Tensor, "B I ..."]:
    """Fallback option"""
    out = torch.zeros((nbatch, *nblocks, *block_size), device=x.device, dtype=x.dtype)
    # Python implementation
    for batch in range(nbatch):
        for blk in product(*(range(nblk) for nblk in nblocks)):
            blk_slc = tuple(
                slice(iblk * st, iblk * st + blk_sz)
                for iblk, st, blk_sz in zip(blk, stride, block_size)
            )
            out_idx = (batch, *blk)
            in_idx = (batch, *blk_slc)
            if mask is not None:
                out[out_idx] = x[in_idx][mask]
            else:
                out[out_idx] = x[in_idx]
    return out


### Helper Functions ###
def prep_shapes(
    x,
    block_size: tuple,
    stride: Optional[tuple] = None,
    mask: Optional[Bool[Tensor, "..."]] = None,
):
    # Infer some shapes
    ndim = len(block_size)
    im_size = x.shape[-ndim:]
    stride = stride if stride is not None else (1,) * ndim
    nblocks = get_nblocks(im_size, block_size, stride)

    # Add or infer batch dim
    batch_shape = x.shape[:-ndim]
    if ndim < len(x.shape):
        x_flat = x.flatten(0, len(x.shape) - ndim - 1)
    else:
        x_flat = x[None]
    nbatch = x_flat.shape[0]

    # Handle mask
    if mask is not None:
        if block_size != mask.shape:
            raise ValueError(
                f"Mask must have same shape as blocks but got mask: {mask.shape} and block_size: {block_size}"
            )
        block_size = (torch.sum(mask).item(),)

    return x_flat, {
        "ndim": ndim,
        "im_size": im_size,
        "stride": stride,
        "nblocks": nblocks,
        "nbatch": nbatch,
        "batch_shape": batch_shape,
        "mask": mask,
        "block_size": block_size,
    }


def get_nblocks(
    im_size: tuple[int, ...],
    block_size: tuple[int, ...],
    block_stride: Optional[tuple[int, ...]] = None,
) -> tuple[int, ...]:
    """Given an image and a block size, returns the number of valid blocks in each direction.

    Blocks may overlap

    Examples
    --------
    >>> get_nblocks((5, 5), (3, 3), (1, 1))
    (3, 3)
    >>> get_nblocks((5, 5), (3, 3), (2, 2))
    (2, 2)
    >>> get_nblocks((6, 6), (3, 3), (2, 2))
    (2, 2)
    >>> get_nblocks((7, 7), (3, 3), (2, 2))
    (3, 3)
    >>> get_nblocks((10, 10), (8, 8), (4, 4))
    (1, 1)
    """
    assert len(im_size) == len(
        block_size
    ), f"im_size {im_size} and block_size {block_size} don't match"
    block_stride = block_stride if block_stride is not None else (1,) * len(block_size)
    output = tuple(
        (im - bl) // st + 1 for im, bl, st in zip(im_size, block_size, block_stride)
    )
    return output
