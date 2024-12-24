"""Functions for interoperability between cupy and pytorch."""

from torch import Tensor

from warnings import warn
import torch
import numpy as np

try:
    import cupy as cp

    cupy_enabled = True
except ImportError as e:
    warn("Cupy not available")
    import numpy as cp

    cupy_enabled = False


__all__ = ["to_pytorch", "from_pytorch", "get_device"]


def get_device(arr) -> torch.device:
    """Bootleg version of sigpy's more-robust get_device"""
    if isinstance(arr, np.ndarray):
        return torch.device("cpu")
    elif cupy_enabled and isinstance(arr, cp.ndarray):
        return torch.device(f"cuda:{arr.device.id}")
    elif isinstance(arr, Tensor):
        return arr.device
    else:
        raise ValueError(f"Can't get device of array type: {type(arr)}")


def to_pytorch(array, requires_grad: bool = False):
    """Zero-copy conversion from numpy/cupy array to pytorch tensor.

    For complex array input, returns a tensor with shape + [2],
    where tensor[..., 0] and tensor[..., 1] represent the real
    and imaginary.

    Args:
        array (numpy/cupy array): input.
        requires_grad(bool): Set .requires_grad output tensor
    Returns:
        PyTorch tensor.

    """
    import torch
    from torch.utils.dlpack import from_dlpack

    device = get_device(array)

    if device.type == "cpu":
        tensor = torch.from_numpy(array)
    else:
        tensor = from_dlpack(array.toDlpack())

    tensor.requires_grad = requires_grad
    return tensor.contiguous()


def from_pytorch(tensor):  # pragma: no cover
    """Zero-copy conversion from pytorch tensor to numpy/cupy array.

    If iscomplex, then tensor must have the last dimension as 2,
    and the output will be viewed as a complex valued array.

    Args:
        tensor (PyTorch tensor): input.
        iscomplex (bool): whether input represents complex valued tensor.

    Returns:
        Numpy/cupy array.

    """
    from torch.utils.dlpack import to_dlpack

    device = tensor.device
    if device.type == "cpu":
        output = tensor.detach().contiguous().numpy()
    else:
        if cupy_enabled:
            output = cp.from_dlpack(to_dlpack(tensor.contiguous()))
        else:
            raise TypeError(
                "CuPy not installed, " "but trying to convert GPU PyTorch Tensor."
            )

    # No longer necessary
    # if iscomplex:
    #     if output.shape[-1] != 2:
    #         raise ValueError('shape[-1] must be 2 when iscomplex is '
    #                          'specified, but got {}'.format(output.shape))

    #     with backend.get_device(output):
    #         if output.dtype == np.float32:
    #             output = output.view(np.complex64)
    #         elif output.dtype == np.float64:
    #             output = output.view(np.complex128)

    #         output = output.reshape(output.shape[:-1])

    return output


def to_pytorch_function(linop):  # pragma: no cover
    """Convert SigPy Linop to PyTorch Function.

    The returned function can be treated as a native
    pytorch function performing the linop operator.
    The function can be backpropagated, applied on GPU arrays,
    and has minimal overhead as the underlying arrays
    are shared without copying.
    For complex valued input/output, the appropriate options
    should be set when calling the function.

    Args:
        linop (Linop): linear operator to be converted.
        input_iscomplex (bool): whether the PyTorch input
            represents complex tensor.
        output_iscomplex (bool): whether the PyTorch output
            represents complex tensor.

    Returns:
        torch.autograd.Function: equivalent PyTorch Function.

    """
    import torch

    class LinopFunction(torch.autograd.Function):
        @staticmethod
        def forward(input):
            return to_pytorch(linop(from_pytorch(input)), input.requires_grad)

        @staticmethod
        def setup_context(ctx, inputs, output):
            pass

        @staticmethod
        def backward(ctx, grad_output):
            return to_pytorch(linop.H(from_pytorch(grad_output)))

        to_nested_mapping = linop

    return LinopFunction
