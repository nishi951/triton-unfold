"""
Notational conventions
block_size: size of each block
stride: step by which each block advances
nblocks: number of blocks along a dimension
"""

from typing import Optional
from itertools import product
from math import prod, ceil

import triton
import triton.language as tl
import torch

__all__ = ["_unfold1d", "_fold1d"]


def unfold(
    x,
    block_size: tuple,
    stride: tuple,
):
    """Cube-like unfolding"""
    # Infer some shapes
    ndim = len(block_size)
    im_size = x.shape[-ndim:]
    stride = stride if stride is not None else (1,) * ndim
    nblocks = get_nblocks(im_size, block_size, stride)

    # Add or infer batch dim
    x_batch_shape = x.shape[:-ndim]
    if ndim < len(x.shape):
        x = x.flatten(0, len(x.shape) - ndim - 1)
    else:
        x = x[None]

    n_batch = x.shape[0]

    with torch.cuda.device(x.device):
        # Allocate output
        y = torch.zeros(
            n_batch,
            *nblocks,
            *block_size,
            device=x.device,
            dtype=x.dtype,
        )
        # grid = (triton.cdiv(n_batch * prod(nblocks), 1),)
        # n_blocks_per_block = X_BLOCK_SIZE // block_size
        # num grids to launch = ceil(nblocks/ n_blocks_per_block
        grid = lambda meta: (
            n_batch * triton.cdiv(nblocks[0], meta["x_blocks_per_block"]),
        )
        X_BLOCK_SIZE = triton.next_power_of_2(block_size[0])
        UNFOLD[ndim][grid](
            x,
            y,
            n_batch,
            *nblocks,
            *block_size,
            *im_size,
            *stride,
            X_BLOCK_SIZE=X_BLOCK_SIZE,
        )

    # Reshape and return
    return y.reshape(*x_batch_shape, *nblocks, *block_size)


def get_configs():
    warps = [1, 2, 4, 8, 16]
    stages = [1, 2, 3, 4, 5]
    block_sizes = [2**i for i in range(8)]
    return [
        triton.Config(
            {"x_blocks_per_block": block_size}, num_warps=warp, num_stages=stages
        )
        for (block_size, warp, stages) in product(block_sizes, warps, stages)
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
    n_batch: int,
    # Number of blocks
    x_nblocks: int,
    # Size of each block
    x_block_dim: int,
    # Size of the input data
    x_size: int,
    # Stride of the blocks
    x_stride: int,
    x_blocks_per_block: int,
    # Size of the triton block (power of 2)
    X_BLOCK_SIZE: tl.constexpr,
):
    # x_blocks_per_block = X_BLOCK_SIZE // x_block_dim

    pid_0 = tl.program_id(0)
    # Batch index, Block index
    NBx = pid_0 * x_blocks_per_block
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
        shape=(n_batch, x_size),
        strides=(1, 1),
        offsets=(0, 0),
        block_shape=(1, X_BLOCK_SIZE),
        order=(0, 1),
    )
    blk_range = tl.arange(0, X_BLOCK_SIZE)
    blk_mask = blk_range < x_block_dim
    # out_offset = N * x_nblocks * x_block_dim + Bx * x_block_dim

    for i in range(x_blocks_per_block):
        if Bx + i < x_nblocks:
            blk = tl.load(in_blk_ptr)
            # Save block to output
            out_offset = N * x_nblocks * x_block_dim + (Bx + i) * x_block_dim
            out_range = blk_range[None, :]
            out_mask = blk_mask[None, :]
            tl.store(out_ptr + out_offset + out_range, blk, out_mask)
        in_blk_ptr = tl.advance(in_blk_ptr, (0, x_stride))


# @triton.autotune(
#     configs=get_configs(),
#     key=["x_nblocks", "x_block_dim", "x_size", "x_stride"],
# )
@triton.jit
def _unfold1d_blockwise(
    in_ptr,
    out_ptr,
    # Number of batches
    n_batch: int,
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
    # Batch index, Block index
    N, Bx = pid_0 // x_nblocks, pid_0 % x_nblocks
    # Get block from input
    x_load_range = tl.arange(0, X_BLOCK_SIZE) + x_stride * Bx
    x_load_mask = x_load_range < x_size
    load_range = x_load_range
    load_mask = x_load_mask
    size = x_size
    blk = tl.load(in_ptr + N * size + load_range, mask=load_mask)

    # Save block to output
    x_store_range = tl.arange(0, X_BLOCK_SIZE)
    x_mask = x_store_range < x_block_dim
    store_range = x_store_range
    store_mask = x_mask
    nblocks = x_nblocks
    block_dim = x_block_dim
    tl.store(
        out_ptr + N * nblocks * block_dim + Bx * block_dim + store_range,
        blk,
        store_mask,
    )


@triton.autotune(
    configs=get_configs(),
    key=["x_nblocks", "x_block_dim", "x_size", "x_stride"],
)
@triton.jit
def _unfold1d_slow(
    in_ptr,
    out_ptr,
    # Number of batches
    n_batch: int,
    # Number of blocks
    x_nblocks: int,
    # Size of each block
    x_block_dim: int,
    # Size of the input data
    x_size: int,
    # Stride of the blocks
    x_stride: int,
    # Size of the triton block (power of 2)
    # X_BLOCK_SIZE: tl.constexpr,
):
    pid_0 = tl.program_id(0)
    # Batch index, Input block location
    N, Inx = pid_0 // x_size, pid_0 % x_size
    if N < n_batch:
        block_dim_per_stride = x_block_dim // x_stride
        x = tl.load(in_ptr + N * x_size + Inx)
        # Block index, Index within block
        Bx_start = Inx // x_stride
        Outx_start = Inx % x_stride

        for i in range(block_dim_per_stride):
            Bx = Bx_start - i
            Outx = Outx_start + i * x_stride

            if Bx >= 0 and Bx < x_nblocks:
                if Outx >= 0 and Outx < x_block_dim:
                    tl.store(
                        out_ptr + N * x_nblocks * x_block_dim + Bx * x_block_dim + Outx,
                        x,
                    )


@triton.autotune(
    configs=get_configs(),
    key=["x_nblocks", "x_block_dim", "x_size", "x_stride"],
)
@triton.jit
def _fold1d(
    in_ptr,
    out_ptr,
    n_batch: int,
    x_nblocks: int,
    x_block_dim: int,
    x_size: int,
    x_stride: int,
    X_BLOCK_SIZE: tl.constexpr,
):
    """
    __________
    |. .| - - ||
    |. . .
    |. . .
    |

    """
    pid_0 = tl.program_id(0)
    N, Outx = pid_0 // x_size, pid_0 % x_size

    if Outx < x_size:
        output = 0.0
        Bx_start = pid_0 // x_stride
        n_strides = x_block_dim // x_stride
        for b in range(n_strides):
            Bx = Bx_start - b
            if Bx >= 0 and Bx < x_nblocks:
                batch_offset = N * x_nblocks * x_block_dim
                block_offset = Bx * x_block_dim
                Inx = Outx - Bx * x_stride

                # Load and accumulate
                x = tl.load(in_ptr + batch_offset + block_offset + Inx)
                output += x

        tl.store(out_ptr + N * x_size + Outx, output)


UNFOLD = {1: _unfold1d}
FOLD = {1: _fold1d}


@triton.jit
def ceildiv(a, b):
    return -(a // -b)


@triton.jit
def floordiv(a, b):
    """This is what // does by default"""
    return a // b


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
