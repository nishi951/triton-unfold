import pytest

from math import prod

import torch
import sigpy as sp

from fold import fold
from utils import from_pytorch, to_pytorch
from nblocks import get_nblocks


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
        "tiny2d",
        "small2d",
        "medium2d",
        pytest.param("large2d", marks=PYTEST_GPU_MARKS),
    ],
)
def test_fold(dev, dtype, spec, request):
    spec = request.getfixturevalue(spec)
    device = torch.device(dev)
    dtype = torch.complex64 if dtype == "complex" else torch.float32

    spec["nblocks"] = get_nblocks(spec["shape"], spec["block_size"], spec["stride"])

    ishape = (*spec["N"], *spec["nblocks"], *spec["block_size"])
    oshape = (*spec["N"], *spec["shape"])

    x = torch.arange(prod(ishape)).reshape(ishape)
    # x = torch.ones(prod(ishape)).reshape(ishape)

    x = x.to(device).to(dtype)

    y_th = fold(x, spec["shape"], spec["block_size"], spec["stride"], spec.get("mask"))

    x = from_pytorch(x)
    y_sp = sp.blocks_to_array(x, oshape, spec["block_size"], spec["stride"])
    assert torch.allclose(y_th, to_pytorch(y_sp))
