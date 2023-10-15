from jax import numpy as jnp

from unrljax import types as t


TRANSPOSE_LAST2_CACHED = [
    [same for same in range(n-2)] + [switched for switched in range(n-1, n-3, -1)]
    for n in range(10)
]


def transpose_last2(a: t.Array) -> t.Array:
    """Transpose the last two axes of a 2- or higher-dimensional matrix"""
    n = len(a.shape)
    assert n >= 2, f'Can only transpose 2- or higher-dimensional matrices, received {n=}.'
    return jnp.transpose(a, TRANSPOSE_LAST2_CACHED[n])
