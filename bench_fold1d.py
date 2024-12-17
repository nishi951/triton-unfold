import torch

from unfold1d_elementwise import unfold, fold
from unfold_native import unfold_torch
import sigpy as sp
import cupy as cp
import numpy as np

from utils.benchmark import benchmark
from utils import Indenter, device_ordinal
from utils import from_pytorch, to_pytorch


def main():
    device = torch.device("cuda:0")
    N = 9
    Nx = 40000

    block_dim = (16,)
    stride = (1,)

    ### Triton Version ###
    def random_unfold_triton():
        x = torch.randn((N, Nx), device=device)
        return unfold(x, block_dim, stride)

    triton_res, _ = benchmark(random_unfold_triton)
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
            x = xp.random.randn(N, Nx)
            return sp.array_to_blocks(x, block_dim, stride)

    sp_res, _ = benchmark(random_unfold_sp)
    summarize(sp_res, "sp")

    # Test correctness
    x = torch.randn((N, Nx), device=device)
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
            indent.print(f"Max Time: {np.max(benchmark_result['timings_ms']):0.3f} ms")
            indent.print(f"Memory: {benchmark_result['max_mem_bytes']} bytes")


if __name__ == "__main__":
    main()
