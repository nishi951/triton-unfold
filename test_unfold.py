import pytest

from math import prod

import torch
import sigpy as sp

from unfold import unfold
from utils import from_pytorch, to_pytorch

# Small, large x 1d, 2d, 3d
# torch vs triton vs sigpy
# adjoint tests for triton and sigpy

PYTEST_GPU_MARKS = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available(), reason="GPU is required but not available"
    ),
]


@pytest.mark.parametrize("dev", ["cpu", pytest.param("cuda", marks=PYTEST_GPU_MARKS)])
@pytest.mark.parametrize("dtype", ["real", "complex"])
@pytest.mark.parametrize(
    "spec",
    [
        "small1d",
        "medium1d",
        pytest.param("large1d", marks=PYTEST_GPU_MARKS),
        "small2d",
        pytest.param("medium2d", marks=PYTEST_GPU_MARKS),
        pytest.param("large2d", marks=PYTEST_GPU_MARKS),
        pytest.param("small3d", marks=PYTEST_GPU_MARKS),
        pytest.param("medium3d", marks=PYTEST_GPU_MARKS),
        pytest.param("large3d", marks=PYTEST_GPU_MARKS),
    ],
)
def test_unfold(dev, dtype, spec, request):
    spec = request.getfixturevalue(spec)
    device = torch.device(dev)
    dtype = torch.complex64 if dtype == "complex" else torch.float32

    ishape = (*spec["N"], *spec["shape"])
    x = torch.arange(prod(ishape)).reshape(ishape)
    # x = torch.ones(prod(ishape)).reshape(ishape)
    x = x.to(device).to(dtype)

    y_th = unfold(x, spec["block_size"], spec["stride"])

    x = from_pytorch(x)
    y_sp = sp.array_to_blocks(x, spec["block_size"], spec["stride"])
    assert torch.allclose(y_th, to_pytorch(y_sp))
