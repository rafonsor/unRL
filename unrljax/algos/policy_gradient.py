from enum import Enum

from jax import numpy as jnp

import unrljax.types as t
from unrljax.objects import Trajectory


class ModelProtocol(t.Protocol):
    def grad(self):
        ...

    def gradlog(self):
        ...

    def step(self):
        ...


class Model(ModelProtocol):
    grad_fun: t.Callable

    def grad(self):
        ...

    def gradlog(self):
        ...

    def step(self):
        ...


class GradientAccumulationMode(Enum):
    EPISODE = 'episode'
    BATCH = 'batch'


class Reinforce:
    policy: "model"

    def __init__(self, learning_rate: float, discount_factor: float):
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate

    def optimise(self, episode: Trajectory):
        for offset, (state, action, reward) in enumerate(episode[::-1]):
            t = len(episode) - offset - 1
            discount = jnp.cumprod(jnp.ones((offset, )) * self.discount_factor)
            future_rewards = [r for _, _, r in episode[t:]]  # todo: must be from t+1 to T
            G = (discount * future_rewards).sum()

            grad = self.policy.gradlog(state.representation, action.representation)
            self.policy.step(self.learning_rate * self.discount_factor ** t * G * grad)

    def batch_optimise(self, episodes: t.Sequence[Trajectory], mode: GradientAccumulationMode = GradientAccumulationMode.BATCH):
        delta = jnp.zeros_like(self.policy)
        n = 0
        for episode in episodes:
            for offset, (state, action, reward) in enumerate(episode[::-1]):
                t = len(episode) - 1 - offset
                discount = jnp.cumprod(jnp.ones((offset, )) * self.discount_factor)
                future_rewards = [r for _, _, r in episode[t:]]  # `reward` already refers to t+1
                G = (discount * future_rewards).sum()
                grad = self.policy.gradlog(state.representation, action.representation)
                delta += self.learning_rate * self.discount_factor ** t * G * grad
                n += 1
            if mode == GradientAccumulationMode.EPISODE:
                self.policy.step(delta / n)
                delta *= 0
                n = 0
        self.policy.step(delta / n)
