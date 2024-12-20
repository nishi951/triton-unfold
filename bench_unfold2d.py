import torch

from unfold import unfold
import sigpy as sp
import cupy as cp
import numpy as np

from utils.benchmark import benchmark
from utils import Indenter, device_ordinal
from utils import from_pytorch, to_pytorch


def main():
    device = torch.device("cuda:0")
    N = 2
    Nx = 100
    Ny = 200

    block_dim = (11, 12)
    stride = (1, 2)
    dtype = torch.complex64
    cp_complex: bool = True

    ### Triton Version ###
    def random_unfold_triton():
        x = torch.randn(N * Nx * Ny, dtype=dtype, device=device).reshape(N, Nx, Ny)
        # x = torch.randn((N, Nx), device=device)
        return unfold(x, block_dim, stride)

    triton_res, _ = benchmark(random_unfold_triton, num_iters=100)
    summarize(triton_res, "triton")

    ### torch.compile Version ###
    # Mildly slower... also takes forever to compile
    # def random_unfold_torch():
    #     x = torch.randn((N, Nx), device=device)
    #     return unfold_torch(x, block_dim, stride)

    # torch_res, _ = benchmark(random_unfold_torch)
    # summarize(torch_res, "torch.compile")

    ### Cupy Version ###
    dev = sp.Device(device_ordinal(device))

    def random_unfold_sp():
        # x = torch.randn((N, Nx), device=device)
        # x = from_pytorch(x)
        xp = dev.xp
        with dev:
            if cp_complex:
                x = xp.random.randn(N, Nx, Ny, dtype=np.float32) + 1j * xp.random.randn(
                    N, Nx, Ny, dtype=np.float32
                )
            else:
                x = xp.random.randn(N, Nx, Ny)
            return sp.array_to_blocks(x, block_dim, stride)

    sp_res, _ = benchmark(random_unfold_sp, num_iters=100)
    summarize(sp_res, "sp")

    # Test correctness
    x = torch.randn((N, Nx, Ny), device=device)
    Bx_triton = unfold(x, block_dim, stride)
    Bx_sp = sp.array_to_blocks(from_pytorch(x), block_dim, stride)
    assert torch.allclose(Bx_triton, to_pytorch(Bx_sp))
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
