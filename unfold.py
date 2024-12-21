# ruff: noqa: F722
from typing import Optional
from jaxtyping import Shaped, Bool, Float
from torch import Tensor

from itertools import product

import triton
import triton.language as tl
import torch

from nblocks import get_nblocks

__all__ = ["unfold"]


def unfold(
    x,
    block_size: tuple,
    stride: Optional[tuple] = None,
    mask: Optional[Bool[Tensor, "..."]] = None,
) -> Tensor:
    """Wrapper that dispatches complex and real tensors"""
    x_flat, shapes = prep_unfold_shapes(x, block_size, stride, mask)
    if torch.is_complex(x_flat):
        x_flat = torch.view_as_real(x_flat)
        y_flat = _unfold(x_flat, **shapes)
        y = y_flat.reshape(
            *shapes["batch_shape"],
            *shapes["nblocks"],
            *shapes["block_size"],
        )
        y = y.reshape(*y.shape[:-1], y.shape[-1] // 2, 2)
        y = torch.view_as_complex(y)
    else:
        y_flat = _unfold(x_flat, **shapes)
        y = y_flat.reshape(
            *shapes["batch_shape"],
            *shapes["nblocks"],
            *shapes["block_size"],
        )
    if mask is not None:
        y = y[..., mask]
    return y


def _unfold(
    x: Shaped[Tensor, "B ..."],
    block_size: tuple[int, ...],
    stride: tuple[int, ...],
    ndim: int,
    im_size: tuple[int, ...],
    nblocks: tuple[int, ...],
    nbatch: int,
    # batch_shape: tuple[int, ...],
    # mask: Bool[Tensor, "..."],
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
            # DEBUG
            # grid = (nblocks[0], nblocks[1])

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
        y = _unfold_torch(x, **shapes)
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


def _get_configs(ndim: int):
    warps = [1, 2]
    stages = [1, 2]
    blocks_per_grid = [[2**i for i in range(4)] for _ in range(ndim)]
    bpg_iter = product(*blocks_per_grid)
    return [
        triton.Config(kwargs=_bpg2dict(*bpg), num_warps=warp, num_stages=stages)
        for (bpg, warp, stages) in product(bpg_iter, warps, stages)
    ]


def _bpg2dict(*bpg):
    out = {}
    for b, n in zip(bpg, ["x", "y", "z"][: len(bpg)]):
        out[f"{n}_blocks_per_grid"] = b
    return out


@triton.autotune(
    configs=_get_configs(ndim=1),
    key=["x_block_dim", "x_size", "x_stride"],
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
    """
    Note: Cannot use make_block_ptr for out_ptr because the output block
    might require masking.
    """
    pid_0 = tl.program_id(0)
    # Batch index, Block index
    NBx = pid_0 * x_blocks_per_grid
    N, Bx = NBx // x_nblocks, NBx % x_nblocks

    in_size = x_size
    nblocks = x_nblocks
    block_dim = x_block_dim
    # Get block from input
    # x_load_range = tl.arange(0, X_BLOCK_SIZE)
    # x_load_mask = x_load_range < x_size
    # load_range = x_load_range
    # load_mask = x_load_mask
    # size = x_size
    # in_offset = N * x_size + Bx * x_stride

    in_blk_ptr = tl.make_block_ptr(
        in_ptr,
        shape=(nbatch, x_size),
        strides=(in_size, 1),
        offsets=(N, Bx * x_stride),
        block_shape=(1, X_BLOCK_SIZE),
        order=(0, 1),
    )
    x_range = tl.arange(0, X_BLOCK_SIZE)
    x_mask = x_range < x_block_dim
    blk_range = x_range
    blk_mask = x_mask
    # out_offset = N * x_nblocks * x_block_dim + Bx * x_block_dim

    out_range = blk_range[None, :]
    out_mask = blk_mask[None, :]

    for i in range(x_blocks_per_grid):
        if Bx + i < x_nblocks:
            blk = tl.load(in_blk_ptr)
            # Save block to output
            out_offset = N * nblocks * block_dim + (Bx + i) * block_dim
            tl.store(out_ptr + out_offset + out_range, blk, out_mask)
        in_blk_ptr = tl.advance(in_blk_ptr, (0, x_stride))


@triton.autotune(
    configs=_get_configs(ndim=2),
    key=[
        "x_block_dim",
        "x_size",
        "x_stride",
        "y_block_dim",
        "y_size",
        "y_stride",
    ],
)
@triton.jit
def _unfold2d(
    in_ptr,
    out_ptr,
    # Number of batches
    nbatch: int,
    # Number of blocks
    x_nblocks: int,
    y_nblocks: int,
    # Size of each block
    x_block_dim: int,
    y_block_dim: int,
    # Size of the input data
    x_size: int,
    y_size: int,
    # Stride of the blocks
    x_stride: int,
    y_stride: int,
    # Size of the triton block (power of 2)
    X_BLOCK_SIZE: tl.constexpr,
    Y_BLOCK_SIZE: tl.constexpr,
    # Number of blocks per grid pid
    x_blocks_per_grid: int,
    y_blocks_per_grid: int,
):
    pid_0 = tl.program_id(0)
    pid_1 = tl.program_id(1)
    # Batch index, Block index
    N = 0
    NBx = pid_0 * x_blocks_per_grid
    N, Bx = NBx // x_nblocks, NBx % x_nblocks
    By = pid_1 * y_blocks_per_grid

    # global sizes
    in_size = x_size * y_size
    nblocks = x_nblocks * y_nblocks
    block_dim = x_block_dim * y_block_dim

    # Get block from input
    # in_offset = N * in_size + Bx * x_stride * y_size + By * y_stride

    in_blk_ptr = tl.make_block_ptr(
        in_ptr,
        shape=(nbatch, x_size, y_size),
        strides=(in_size, y_size, 1),
        offsets=(N, Bx * x_stride, By * y_stride),
        block_shape=(1, X_BLOCK_SIZE, Y_BLOCK_SIZE),
        order=(0, 1, 2),
    )
    x_range = tl.arange(0, X_BLOCK_SIZE)
    y_range = tl.arange(0, Y_BLOCK_SIZE)
    x_mask = x_range < x_block_dim
    y_mask = y_range < y_block_dim
    blk_range = x_range[:, None] * y_block_dim + y_range[None, :]
    blk_mask = x_mask[:, None] & y_mask[None, :]
    # out_offset = N * x_nblocks * x_block_dim + Bx * x_block_dim
    # add batch dim
    out_range = blk_range[None, :, :]
    out_mask = blk_mask[None, :, :]

    for i in range(x_blocks_per_grid):
        if Bx + i < x_nblocks:
            x_blk_offset = (Bx + i) * y_nblocks * block_dim
            in_blk_ptr_x = in_blk_ptr
            for j in range(y_blocks_per_grid):
                if By + j < y_nblocks:
                    y_blk_offset = (By + j) * block_dim
                    blk = tl.load(in_blk_ptr)
                    out_offset = N * nblocks * block_dim + x_blk_offset + y_blk_offset
                    tl.store(out_ptr + out_offset + out_range, blk, out_mask)
                in_blk_ptr = tl.advance(in_blk_ptr, (0, 0, y_stride))
            in_blk_ptr = in_blk_ptr_x
        in_blk_ptr = tl.advance(in_blk_ptr, (0, x_stride, 0))


@triton.autotune(
    configs=_get_configs(ndim=3),
    key=[
        "x_block_dim",
        "x_size",
        "x_stride",
        "y_block_dim",
        "y_size",
        "y_stride",
        "z_block_dim",
        "z_size",
        "z_stride",
    ],
)
@triton.jit
def _unfold3d(
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
    """TODO: Not finished"""
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


UNFOLD = {1: _unfold1d, 2: _unfold2d}


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
            out[out_idx] = x[in_idx]
    return out


def prep_unfold_shapes(
    x,
    block_size: tuple,
    stride: Optional[tuple] = None,
    mask: Optional[Bool[Tensor, "..."]] = None,
):
    # Infer some shapes
    ndim = len(block_size)
    im_size = x.shape[-ndim:]
    stride = stride if stride is not None else (1,) * ndim

    if torch.is_complex(x):
        block_size = list(block_size)
        block_size[-1] *= 2
        stride = list(stride)
        stride[-1] *= 2

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
        # if torch.is_complex(x):
        #     mask = torch.repeat_interleave(mask, repeats=2, dim=-1).contiguous()
        # if block_size != mask.shape:
        #     raise ValueError(
        #         f"Mask must have same shape as blocks but got mask: {mask.shape} and block_size: {block_size}"
        #     )
        # block_size = (int(torch.sum(mask).item()),)
        mask = mask.to(x.device)

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
