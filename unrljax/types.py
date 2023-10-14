from typing import *  # noqa

from jax import Array

IntArray = Array  # expected dtype == jnp.int32
FloatArray = Array  # expected dtype == jnp.float32

IntLike = IntArray | int
FloatLike = FloatArray | float
