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
        x_flat = torch.flatten(x_flat, -2, -1)  # Flatten real/imag into last dim
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
    if x.is_cuda and ndim in FOLD.keys():
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
        y = _fold_torch(x, block_size, stride, ndim, im_size, nblocks, nbatch)
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
            x_Lpad = Bx * x_stride - x_lower
            # Rpad = x_upper - Bx * x_stride + x_block_dim
            x_in_range = tl.arange(0, X_BLOCK_SIZE)
            # x_in_range = x_in_range - x_Lpad
            x_in_mask = ((x_in_range - x_Lpad) >= 0) & (
                (x_in_range - x_Lpad) < x_block_dim
            )
            # Load block
            block_offset = Bx * block_dim
            block_offset = block_offset - x_Lpad
            in_range = x_in_range
            in_mask = x_in_mask
            blk = tl.load(in_ptr + in_offset + block_offset + in_range, in_mask)
            output += blk
    tl.store(out_ptr + out_offset + out_range, output, out_mask)


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
def _fold2d(
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
):
    pid_0 = tl.program_id(0)
    pid_1 = tl.program_id(1)
    # x_blocks_per_batch = tl.ceil(x_size / X_BLOCK_SIZE)
    x_blocks_per_batch = cdiv(x_size, X_BLOCK_SIZE)
    y_blocks_per_batch = cdiv(y_size, Y_BLOCK_SIZE)

    # Batch index, Block index
    N, Ix = pid_0 // x_blocks_per_batch, pid_0 % x_blocks_per_batch
    Iy = pid_1 % y_blocks_per_batch

    nblocks = x_nblocks * y_nblocks
    block_dim = x_block_dim * y_block_dim
    size = x_size * y_size

    in_offset = N * nblocks * block_dim

    # Find overlapping blocks with range
    x_lower = Ix * X_BLOCK_SIZE
    x_upper = x_lower + X_BLOCK_SIZE
    Bx_lower = cdiv(x_lower - x_block_dim + 1, x_stride)
    Bx_upper = cdiv(x_upper, x_stride)  # non-inclusive
    y_lower = Iy * Y_BLOCK_SIZE
    y_upper = y_lower + Y_BLOCK_SIZE
    By_lower = cdiv(y_lower - y_block_dim + 1, y_stride)
    By_upper = cdiv(y_upper, y_stride)  # non-inclusive

    # Initialize output
    output = tl.zeros((1, X_BLOCK_SIZE, Y_BLOCK_SIZE), tl.float32)
    x_range = tl.arange(0, X_BLOCK_SIZE) + x_lower
    x_mask = x_range < x_size
    y_range = tl.arange(0, Y_BLOCK_SIZE) + y_lower
    y_mask = y_range < y_size

    out_offset = N * size
    out_range = x_range[:, None] * y_size + y_range[None, :]
    out_mask = x_mask[:, None] & y_mask[None, :]

    out_range = out_range[None, :]
    out_mask = out_mask[None, :]

    for Bx in range(Bx_lower, Bx_upper):
        if Bx >= 0 and Bx < x_nblocks:
            x_Lpad = Bx * x_stride - x_lower
            # Rpad = x_upper - Bx * x_stride + x_block_dim
            x_in_range = tl.arange(0, X_BLOCK_SIZE)
            x_in_mask = ((x_in_range - x_Lpad) >= 0) & (
                (x_in_range - x_Lpad) < x_block_dim
            )
            for By in range(By_lower, By_upper):
                if By >= 0 and By < y_nblocks:
                    y_Lpad = By * y_stride - y_lower
                    # Rpad = x_upper - Bx * x_stride + x_block_dim
                    y_in_range = tl.arange(0, Y_BLOCK_SIZE)
                    y_in_mask = ((y_in_range - y_Lpad) >= 0) & (
                        (y_in_range - y_Lpad) < y_block_dim
                    )
                    # Load block
                    block_offset = Bx * y_nblocks * block_dim + By * block_dim
                    block_offset = block_offset - (x_Lpad * y_block_dim + y_Lpad)
                    in_range = x_in_range[:, None] * y_block_dim + y_in_range[None, :]
                    in_mask = x_in_mask[:, None] & y_in_mask[None, :]
                    blk = tl.load(in_ptr + in_offset + block_offset + in_range, in_mask)
                    output += blk
    tl.store(out_ptr + out_offset + out_range, output, out_mask)


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
def _fold3d(
    in_ptr,
    out_ptr,
    # Number of batches
    nbatch: int,
    # Number of blocks
    x_nblocks: int,
    y_nblocks: int,
    z_nblocks: int,
    # Size of each block
    x_block_dim: int,
    y_block_dim: int,
    z_block_dim: int,
    # Size of the input data
    x_size: int,
    y_size: int,
    z_size: int,
    # Stride of the blocks
    x_stride: int,
    y_stride: int,
    z_stride: int,
    # Size of the triton block (power of 2)
    X_BLOCK_SIZE: tl.constexpr,
    Y_BLOCK_SIZE: tl.constexpr,
    Z_BLOCK_SIZE: tl.constexpr,
):
    pid_0 = tl.program_id(0)
    pid_1 = tl.program_id(1)
    pid_1 = tl.program_id(2)
    # x_blocks_per_batch = tl.ceil(x_size / X_BLOCK_SIZE)
    x_blocks_per_batch = cdiv(x_size, X_BLOCK_SIZE)
    y_blocks_per_batch = cdiv(y_size, Y_BLOCK_SIZE)
    z_blocks_per_batch = cdiv(z_size, Z_BLOCK_SIZE)

    # Batch index, Block index
    N, Ix = pid_0 // x_blocks_per_batch, pid_0 % x_blocks_per_batch
    Iy = pid_1 % y_blocks_per_batch
    Iz = pid_2 % z_blocks_per_batch

    nblocks = x_nblocks * y_nblocks * z_nblocks
    block_dim = x_block_dim * y_block_dim * z_block_dim
    size = x_size * y_size * z_size

    in_offset = N * nblocks * block_dim

    # Find overlapping blocks with range
    x_lower = Ix * X_BLOCK_SIZE
    x_upper = x_lower + X_BLOCK_SIZE
    Bx_lower = cdiv(x_lower - x_block_dim + 1, x_stride)
    Bx_upper = cdiv(x_upper, x_stride)  # non-inclusive
    y_lower = Iy * Y_BLOCK_SIZE
    y_upper = y_lower + Y_BLOCK_SIZE
    By_lower = cdiv(y_lower - y_block_dim + 1, y_stride)
    By_upper = cdiv(y_upper, y_stride)  # non-inclusive
    z_lower = Iz * Z_BLOCK_SIZE
    z_upper = z_lower + Z_BLOCK_SIZE
    Bz_lower = cdiv(z_lower - z_block_dim + 1, z_stride)
    Bz_upper = cdiv(z_upper, z_stride)  # non-inclusive

    # Initialize output
    output = tl.zeros((1, X_BLOCK_SIZE, Y_BLOCK_SIZE), tl.float32)
    x_range = tl.arange(0, X_BLOCK_SIZE) + x_lower
    x_mask = x_range < x_size
    y_range = tl.arange(0, Y_BLOCK_SIZE) + y_lower
    y_mask = y_range < y_size
    z_range = tl.arange(0, Z_BLOCK_SIZE) + z_lower
    z_mask = z_range < z_size

    out_offset = N * size
    out_range = (
        x_range[:, None, None] * y_size + y_range[None, :, None]
    ) * z_size + z_range[None, None, :]
    out_mask = x_mask[:, None, None] & (y_mask[None, :, None] & z_mask[None, None, :])

    out_range = out_range[None, :, :, :]
    out_mask = out_mask[None, :, :, :]

    for Bx in range(Bx_lower, Bx_upper):
        if Bx >= 0 and Bx < x_nblocks:
            x_Lpad = Bx * x_stride - x_lower
            # Rpad = x_upper - Bx * x_stride + x_block_dim
            x_in_range = tl.arange(0, X_BLOCK_SIZE)
            x_in_mask = ((x_in_range - x_Lpad) >= 0) & (
                (x_in_range - x_Lpad) < x_block_dim
            )
            for By in range(By_lower, By_upper):
                if By >= 0 and By < y_nblocks:
                    y_Lpad = By * y_stride - y_lower
                    # Rpad = x_upper - Bx * x_stride + x_block_dim
                    y_in_range = tl.arange(0, Y_BLOCK_SIZE)
                    y_in_mask = ((y_in_range - y_Lpad) >= 0) & (
                        (y_in_range - y_Lpad) < y_block_dim
                    )
                    for Bz in range(Bz_lower, Bz_upper):
                        if Bz >= 0 and Bz < z_nblocks:
                            z_in_range = tl.arange(0, Z_BLOCK_SIZE)
                            z_in_mask = ((z_in_range - z_Lpad) >= 0) & (
                                (z_in_range - z_Lpad) < z_block_dim
                            )
                            # Load block
                            block_offset = (
                                (Bx * y_nblocks + By) * z_nblocks + Bz
                            ) * block_dim
                            block_offset = block_offset - (
                                (x_Lpad * y_block_dim + y_Lpad) * z_block_dim + z_Lpad
                            )
                            in_range = (
                                x_in_range[:, None, None] * y_block_dim
                                + y_in_range[None, :, None]
                            ) * z_block_dim + z_in_range[None, None, :]
                            in_mask = x_in_mask[:, None, None] & (
                                y_in_mask[None, :, None] & z_in_mask[None, None, :]
                            )
                            blk = tl.load(
                                in_ptr + in_offset + block_offset + in_range, in_mask
                            )
                            output += blk
    tl.store(out_ptr + out_offset + out_range, output, out_mask)


@triton.jit
def cdiv(a, b):
    return tl.cast(tl.ceil(a / b), tl.int32)


FOLD = {1: _fold1d, 2: _fold2d, 3: _fold3d}


def _fold_torch(
    x: Shaped[Tensor, "B ..."],
    block_size: tuple[int, ...],
    stride: tuple[int, ...],
    ndim: int,
    im_size: tuple[int, ...],
    nblocks: tuple[int, ...],
    nbatch: int,
) -> Shaped[Tensor, "B I ..."]:
    """Fallback option

    Note: Compile takes forever
    """
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
    nblocks = get_nblocks(im_size, block_size, stride)
    if torch.is_complex(x):
        im_size = list(im_size)
        im_size[-1] *= 2
        block_size = list(block_size)
        block_size[-1] *= 2
        stride = list(stride)
        stride[-1] *= 2

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
