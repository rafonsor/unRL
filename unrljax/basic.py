
from jax import numpy as jnp
from jax import random

import unrljax.types as t


def softmax(seq: t.Array) -> t.Array:
    """Maximum-shifted Soft-argmax"""
    exps = jnp.exp(seq - seq.max())
    return exps / exps.sum()


def logsoftmax(seq: t.Array) -> t.Array:
    """Maximum-shifted LogSoft-argmax returning log probabilities"""
    shifted = seq - seq.max()
    return shifted - jnp.log(jnp.exp(shifted).sum())


def softmax_temperature(seq: t.Array, tau: t.FloatLike) -> t.Array:
    """Maximum-shifted Soft-argmax with Temperature"""
    if tau <= 0:
        raise ValueError('Temperature must be positive')
    exps = jnp.exp((seq - seq.max()) / tau)
    return exps / exps.sum()


def argmax_all(seq: t.Array) -> t.Array:
    """Return all arguments with maximal value"""
    return jnp.flatnonzero(jnp.isclose(seq, seq.max()))


def argmax_random(key: random.PRNGKey, seq: t.Array) -> int:
    """Return argmax with a random tie among equal maxima"""
    if seq.size and (args := argmax_all(seq)).size:
        return random.choice(key, args).item()
    return -1


def sample_gaussian(key: random.PRNGKey, mu: t.FloatArray, sigma: t.FloatArray) -> t.FloatArray:
    """Gaussian sampling using the Box-Muller transform"""
    u = random.uniform(key, (2, *mu.shape))
    z = jnp.sqrt(-2*jnp.log(u[1])) * jnp.cos(2*jnp.pi*u[2])
    return mu + z*sigma
