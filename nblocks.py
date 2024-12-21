from typing import Optional
from jaxtyping import Bool
from torch import Tensor

import torch

__all__ = ["get_nblocks"]


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
