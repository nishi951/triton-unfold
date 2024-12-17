from collections.abc import Iterable
from torch import Tensor
import torch

__all__ = ["device_ordinal", "same_storage", "idx2dev"]


def device_ordinal(device: torch.device) -> int:
    """Get the integer representing the device ordinal.

    Parameters
    ----------
    device : torch.device
        The device in question.

    Returns
    -------
    int
        The device ordinal - 0+ if GPU, -1 if cpu
    """
    return torch.zeros(1, device=device).get_device()


def same_storage(x: Tensor, y: Tensor):
    """Determine if tensors share the same storage or not"""
    x_ptrs = set(e.data_ptr() for e in x.view(-1))
    y_ptrs = set(e.data_ptr() for e in y.view(-1))
    return (x_ptrs <= y_ptrs) or (y_ptrs <= x_ptrs)


def idx2dev(idx: int | list):
    if isinstance(idx, Iterable):
        return [idx2dev(i) for i in idx]
    return torch.device(f"cuda:{idx}" if int(idx) >= 0 else "cpu")
