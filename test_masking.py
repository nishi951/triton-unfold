import pytest
from math import prod
import torch

from unfold import unfold

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
        # "medium1d",
        # pytest.param("large1d", marks=PYTEST_GPU_MARKS),
        "small2d",
        # pytest.param("medium2d", marks=PYTEST_GPU_MARKS),
        # pytest.param("large2d", marks=PYTEST_GPU_MARKS),
    ],
)
def test_unfold_mask(dev, dtype, spec, request):
    spec = request.getfixturevalue(spec)
    device = torch.device(dev)
    dtype = torch.complex64 if dtype == "complex" else torch.float32

    ishape = (*spec["N"], *spec["shape"])
    x = torch.arange(prod(ishape)).reshape(ishape)
    # x = torch.ones(prod(ishape)).reshape(ishape)
    x = x.to(device).to(dtype)

    mask = spec["mask"]
    y_th = unfold(x, spec["block_size"], spec["stride"])
    y_th_masked = unfold(x, spec["block_size"], spec["stride"], mask)

    assert torch.allclose(y_th[..., mask.to(device)], y_th_masked)
