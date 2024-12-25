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


@pytest.fixture
def small1d():
    spec = {
        "N": (1,),
        "shape": (15,),
        "block_size": (3,),
        "stride": (1,),
    }
    return spec


@pytest.fixture
def medium1d():
    spec = {
        "N": (2, 2),
        "shape": (33,),
        "block_size": (7,),
        "stride": (2,),
    }
    return spec


@pytest.fixture
def large1d():
    spec = {
        "N": (20, 1),
        "shape": (1025,),
        "block_size": (25,),
        "stride": (25,),
    }
    return spec


@pytest.fixture
def small2d():
    spec = {
        "N": (1,),
        "shape": (15, 15),
        "block_size": (3, 3),
        "stride": (1, 1),
    }
    return spec


@pytest.fixture
def medium2d():
    spec = {
        "N": (2, 2),
        "shape": (33, 43),
        "block_size": (7, 7),
        "stride": (2, 2),
    }
    return spec


@pytest.fixture
def large2d():
    spec = {
        "N": (20, 1),
        "shape": (125, 125),
        "block_size": (25, 25),
        "stride": (25, 25),
    }
    return spec


@pytest.mark.parametrize("dev", ["cpu", pytest.param("cuda", marks=PYTEST_GPU_MARKS)])
@pytest.mark.parametrize("dtype", ["real", "complex"])
@pytest.mark.parametrize(
    "spec",
    [
        "small1d",
        "medium1d",
        pytest.param("large1d", marks=PYTEST_GPU_MARKS),
        "small2d",
        "medium2d",
        pytest.param("large2d", marks=PYTEST_GPU_MARKS),
    ],
)
def test_unfold(dev, dtype, spec, request):
    spec = request.getfixturevalue(spec)
    device = torch.device(dev)
    dtype = torch.complex64 if dtype == "complex" else torch.float32

    # x = torch.randn(spec["N"], *spec["shape"], device=device, dtype=dtype)
    x = torch.arange(prod(spec["N"]) * prod(spec["shape"])).reshape(
        *spec["N"], *spec["shape"]
    )
    x = x.to(device).to(dtype)

    y_th = unfold(x, spec["block_size"], spec["stride"], spec.get("mask"))

    x = from_pytorch(x)
    y_sp = sp.array_to_blocks(x, spec["block_size"], spec["stride"])
    assert torch.allclose(y_th, to_pytorch(y_sp))
