import torch

import triton
import triton.language as tl
import numpy as np

from utils.benchmark import benchmark
from utils import Indenter, device_ordinal
from utils import from_pytorch, to_pytorch


@triton.jit
def _unfold1d(
    in_ptr,
    out_ptr,
    nbatch,
    x_size,
    x_stride,
    x_nblocks,
    x_block_dim,
    X_BLOCK_SIZE: tl.constexpr,
):
    in_blk_ptr = tl.make_block_ptr(
        in_ptr,
        (nbatch, x_size),
        strides=(1, 1),
        offsets=(0, 0),
        block_shape=(1, X_BLOCK_SIZE),
        order=(0, 1),
    )
    for Bx in range(x_nblocks):
        blk = tl.load(in_blk_ptr)
        blk_range = tl.arange(0, X_BLOCK_SIZE)
        blk_mask = blk_range < x_block_dim
        offsets = Bx * x_block_dim
        out_offset = blk_range[None, :] + offsets
        blk_mask = blk_mask[None, :]
        tl.store(out_ptr + out_offset, blk, blk_mask)
        in_blk_ptr = tl.advance(in_blk_ptr, (0, x_stride))


def main():
    device = torch.device("cuda:0")
    N = 1
    Nx = 10

    block_dim = (4,)
    stride = (2,)
    x = torch.randn(
        1,
    )
    nblocks = (7,)

    def random_unfold_triton():
        x = torch.arange(N * Nx, device=device).reshape(N, Nx)
        y = torch.zeros(N, *nblocks, *block_dim, device=device)
        grid = (1,)
        _unfold1d[grid](x, y, N, Nx, *stride, *nblocks, *block_dim, X_BLOCK_SIZE=8)
        return x, y

    triton_res, triton_output = benchmark(random_unfold_triton, num_iters=100)
    summarize(triton_res, "triton")
    x, y = triton_output
    print(x)
    print(y)
    breakpoint()


def summarize(benchmark_result, name: str):
    with Indenter() as indent:
        print(name)
        with indent:
            indent.print(
                f"Mean Time: {np.mean(benchmark_result['timings_ms']):0.3f} ms"
            )
            indent.print(f"Min Time: {np.min(benchmark_result['timings_ms']):0.3f} ms")
            indent.print(f"Max Time: {np.max(benchmark_result['timings_ms']):0.3f} ms")
            indent.print(f"Memory: {benchmark_result['max_mem_bytes']} bytes")


if __name__ == "__main__":
    main()
