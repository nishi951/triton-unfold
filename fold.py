#!/usr/bin/env python3


def fold(
    x,
    block_size: tuple,
    stride: tuple,
):
    """Cube-like unfolding"""
    # Infer some shapes
    ndim = len(block_size)
    im_size = x.shape[-ndim:]
    stride = stride if stride is not None else (1,) * ndim
    nblocks = get_nblocks(im_size, block_size, stride)

    # Add or infer batch dim
    x_batch_shape = x.shape[-ndim:]
    if ndim < len(x.shape):
        x = x.flatten(0, len(x.shape) - ndim - 1)
    else:
        x = x[None]

    n_batch = x.shape[0]

    # Call kernel
    y = torch.zeros(n_batch, *nblocks, *block_size)
    grid = (triton.cdiv(n_batch * prod(nblocks) * prod(block_size), 1),)
    FOLD[ndim][grid](
        x,
        y,
        n_batch,
        *nblocks,
        *block_size,
        *im_size,
        *stride,
    )

    # Reshape and return
    return y.reshape(*x_batch_shape, *nblocks, *block_size)
