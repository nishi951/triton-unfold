# ruff: noqa: F722
from typing import Optional
from jaxtyping import Shaped, Bool
from torch import Tensor

from itertools import product

import torch
import triton
import triton.language as tl

from nblocks import get_nblocks

import pdb

__all__ = ["fold"]


def fold(
    x,
    im_size: tuple,
    block_size: tuple,
    stride: tuple,
    mask: Optional[Bool[Tensor, "..."]] = None,
) -> Tensor:
    """Cube-like unfolding"""
    x_flat, shapes = prep_fold_shapes(x, im_size, block_size, stride, mask)
    if mask is not None:
        tmp = torch.zeros(
            shapes["nbatch"],
            *shapes["nblocks"],
            *shapes["block_size"],
            dtype=x.dtype,
            device=x.device,
        )
        tmp[..., mask] = x_flat
        x_flat = tmp

    if torch.is_complex(x_flat):
        x_flat = torch.view_as_real(x_flat)
        y_flat = _fold(x_flat, **shapes)
        y = y_flat.reshape(*shapes["batch_shape"], *shapes["im_size"])
        y = y.reshape(*y.shape[:-1], y.shape[-1] // 2, 2)
        y = torch.view_as_complex(y)
    else:
        y_flat = _fold(x_flat, **shapes)
        y = y_flat.reshape(*shapes["batch_shape"], *shapes["im_size"])
    return y


def _fold(
    x: Shaped[Tensor, "B ..."],
    block_size: tuple[int, ...],
    stride: tuple[int, ...],
    ndim: int,
    im_size: tuple[int, ...],
    nblocks: tuple[int, ...],
    nbatch: int,
    **kwargs,
):
    """Implementation of fold"""
    if x.is_cuda and ndim in (1, 2, 3):
        with torch.cuda.device(x.device):
            # Allocate output
            y = torch.zeros(
                nbatch,
                *im_size,
                device=x.device,
                dtype=x.dtype,
            )
            grid = _get_grid(ndim, nbatch, im_size)

            FOLD[ndim][grid](
                x,
                y,
                nbatch,
                *nblocks,
                *block_size,
                *im_size,
                *stride,
                #    *(8,),
            )
    else:
        y = _fold_torch(x)
    return y


def _get_grid(ndim: int, nbatch, im_size):
    if ndim == 1:
        grid = lambda meta: (  # noqa: E731
            nbatch * triton.cdiv(im_size[0], meta["X_BLOCK_SIZE"]),
        )
    elif ndim == 2:
        grid = lambda meta: (  # noqa: E731
            nbatch * triton.cdiv(im_size[0], meta["X_BLOCK_SIZE"]),
            triton.cdiv(im_size[1], meta["Y_BLOCK_SIZE"]),
        )
    elif ndim == 3:
        grid = lambda meta: (  # noqa: E731
            nbatch * triton.cdiv(im_size[0], meta["X_BLOCK_SIZE"]),
            triton.cdiv(im_size[1], meta["Y_BLOCK_SIZE"]),
            triton.cdiv(im_size[2], meta["Z_BLOCK_SIZE"]),
        )
    else:
        raise ValueError(f"Invalid ndim = {ndim}")
    return grid


def _get_configs(ndim: int):
    warps = [1, 2]
    stages = [1, 2]
    block_sizes = [[2**i for i in range(8)] for _ in range(ndim)]
    bsz_iter = product(*block_sizes)
    return [
        triton.Config(kwargs=_bsz2dict(*bsz), num_warps=warp, num_stages=stages)
        for (bsz, warp, stages) in product(bsz_iter, warps, stages)
    ]


def _bsz2dict(*bsz):
    out = {}
    for b, n in zip(bsz, ["X", "Y", "Z"][: len(bsz)]):
        out[f"{n}_BLOCK_SIZE"] = b
    return out


@triton.autotune(
    configs=_get_configs(ndim=1),
    key=["x_block_dim", "x_size", "x_stride"],
)
@triton.jit
def _fold1d(
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
):
    pid_0 = tl.program_id(0)
    # x_blocks_per_batch = tl.ceil(x_size / X_BLOCK_SIZE)
    x_blocks_per_batch = cdiv(x_size, X_BLOCK_SIZE)

    # Batch index, Block index
    N, Ix = pid_0 // x_blocks_per_batch, pid_0 % x_blocks_per_batch
    # N = pid_0 // x_blocks_per_batch

    nblocks = x_nblocks
    block_dim = x_block_dim
    size = x_size

    in_offset = N * nblocks * block_dim

    # Find overlapping blocks with range
    x_lower = Ix * X_BLOCK_SIZE
    x_upper = x_lower + X_BLOCK_SIZE
    Bx_lower = cdiv(x_lower - x_block_dim + 1, x_stride)
    Bx_upper = cdiv(x_upper, x_stride)  # non-inclusive

    # Initialize output
    output = tl.zeros((1, X_BLOCK_SIZE), tl.float32)
    x_range = tl.arange(0, X_BLOCK_SIZE) + x_lower
    x_mask = x_range < x_size
    out_offset = N * size
    out_range = x_range
    out_mask = x_mask

    out_range = out_range[None, :]
    out_mask = out_mask[None, :]
    # pdb.set_trace()

    for Bx in range(Bx_lower, Bx_upper):
        if Bx >= 0 and Bx < x_nblocks:
            Lpad = Bx * x_stride - x_lower
            # Rpad = x_upper - Bx * x_stride + x_block_dim
            x_in_range = tl.arange(0, X_BLOCK_SIZE) + Bx * x_block_dim
            x_in_range = x_in_range - Lpad
            x_in_mask = (x_in_range >= Bx * x_block_dim) & (
                x_in_range < (Bx + 1) * x_block_dim
            )
            x_in_mask = x_in_mask
            blk = tl.load(in_ptr + in_offset + x_in_range, x_in_mask)
            output += blk
    tl.store(out_ptr + out_offset + out_range, output, out_mask)


@triton.jit
def cdiv(a, b):
    return tl.cast(tl.ceil(a / b), tl.int32)


FOLD = {1: _fold1d}


@torch.compile
def _fold_torch(
    x: Shaped[Tensor, "B ..."],
    block_size: tuple[int, ...],
    stride: tuple[int, ...],
    ndim: int,
    im_size: tuple[int, ...],
    nblocks: tuple[int, ...],
    nbatch: int,
    mask: Bool[Tensor, "..."],
) -> Shaped[Tensor, "B I ..."]:
    """Fallback option"""
    out = torch.zeros((nbatch, *im_size), device=x.device, dtype=x.dtype)
    # Python implementation
    for batch in range(nbatch):
        for blk in product(*(range(nblk) for nblk in nblocks)):
            blk_slc = tuple(
                slice(iblk * st, iblk * st + blk_sz)
                for iblk, st, blk_sz in zip(blk, stride, block_size)
            )
            in_idx = (batch, *blk)
            out_idx = (batch, *blk_slc)
            out[out_idx] += x[in_idx]
    return out


def prep_fold_shapes(x, im_size, block_size, stride, mask):
    ndim = len(block_size)
    stride = stride if stride is not None else (1,) * ndim
    if torch.is_complex(x):
        block_size = list(block_size)
        block_size[-1] *= 2
        stride = list(stride)
        stride[-1] *= 2
    nblocks = get_nblocks(im_size, block_size, stride)

    # Add or infer batch dim
    batch_shape = x.shape[: (-2 * ndim)]
    if 2 * ndim < len(x.shape):
        x_flat = x.flatten(0, len(x.shape) - (2 * ndim) - 1)
    else:
        x_flat = x[None]
    nbatch = x_flat.shape[0]

    # Handle mask
    if mask is not None:
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
