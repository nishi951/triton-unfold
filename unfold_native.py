from typing import Optional
from jaxtyping import Shaped
from torch import Tensor
from unfold1d_elementwise import get_nblocks

from itertools import product

import torch

__all__ = ["unfold_torch"]


@torch.compile
def unfold_torch(
    x: Shaped[Tensor, "B ..."],
    block_size: tuple[int, ...],
    stride: Optional[tuple[int, ...]] = None,
) -> Shaped[Tensor, "B I ..."]:
    nbatch = x.shape[0]
    ndim = len(block_size)
    stride = (1,) * ndim if stride is None else stride
    nblocks = get_nblocks(x.shape[-ndim:], block_size, stride)
    # Flatten batch dimensions
    out = torch.zeros((nbatch, *nblocks, *block_size), device=x.device, dtype=x.dtype)
    # Python implementation
    for batch in range(nbatch):
        for blk in product(*(range(nblk) for nblk in nblocks)):
            blk_slc = tuple(
                slice(iblk * st, iblk * st + blk_sz)
                for iblk, st, blk_sz in zip(blk, stride, block_size)
            )
            out_idx = (batch, *blk)
            in_idx = (batch, *blk_slc)
            out[out_idx] = x[in_idx]
    return out
