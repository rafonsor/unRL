# Copyright 2023 The unRL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from enum import Enum

import jax.numpy as jnp

import unrljax.types as t
from unrljax.objects import Trajectory


class ModelProtocol(t.Protocol):
    def __call__(self, *args, **kwargs) -> t.Any:
        ...

    def grad(self, *args, **kwargs) -> t.Any:
        ...

    def gradlog(self, *args, **kwargs) -> t.Any:
        ...

    def step(self, *args, **kwargs):
        ...


class Model(ModelProtocol):
    _grad_fun: t.Callable

    def __call__(self, *args, **kwargs) -> t.Any:
        ...

    def grad(self, *args, **kwargs) -> t.Any:
        y = self(*args, **kwargs)
        return self._grad_fun(y)

    def gradlog(self, *args, **kwargs) -> t.Any:
        logy = jnp.log(self(*args, **kwargs))
        return self._grad_fun(logy)

    def step(self, *args, **kwargs):
        ...


class GradientAccumulationMode(Enum):
    EPISODE = 'episode'
    BATCH = 'batch'


class Reinforce:
    """REINFORCE Monte-Carlo Policy-Gradient"""
    policy: Model

    def __init__(self, learning_rate: float, discount_factor: float):
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate

    def _calculate_return(self, episode: Trajectory, offset: int, timestep: int) -> float:
        discount = jnp.cumprod(jnp.ones((offset + 1, )) * self.discount_factor)
        future_rewards = [r for _, _, r in episode[timestep - 1:]]  # note transition t ponts to `reward` t+1
        return (discount * future_rewards).sum()

    def _calculate_delta(self, episode: Trajectory, offset: int) -> t.FloatLike:
        timestep = len(episode) - 1 - offset
        (state, action, _) = episode[timestep]
        grad = self.policy.gradlog(state.representation, action.representation)
        G = self._calculate_return(episode, offset, timestep)
        delta_policy = self.learning_rate * self.discount_factor ** timestep * G * grad
        return delta_policy

    def optimise(self, episode: Trajectory):
        for offset in range(len(episode)):
            delta_policy = self._calculate_delta(episode, offset)
            self.policy.step(delta_policy)

    def batch_optimise(self,
                       episodes: t.Sequence[Trajectory],
                       mode: GradientAccumulationMode = GradientAccumulationMode.BATCH):
        delta_policy = None
        n = 0
        for episode in episodes:
            for offset in range(len(episode)):
                if delta_policy is None:
                    delta_policy = self._calculate_delta(episode, offset)
                else:
                    delta_policy += self._calculate_delta(episode, offset)
                n += 1
            if mode == GradientAccumulationMode.EPISODE:
                self.policy.step(delta_policy / n)
                delta_policy *= 0
                n = 0
        self.policy.step(delta_policy / n)


class BaselineReinforce:
    """REINFORCE Monte-Carlo Policy-Gradient with Baseline that learns both a policy and state-values"""
    policy: Model
    state_value: Model

    def __init__(self,
                 learning_rate_policy: float,
                 learning_rate_value: float,
                 discount_factor: float):
        self.discount_factor = discount_factor
        self.learning_rate_policy = learning_rate_policy
        self.learning_rate_value = learning_rate_value

    def _calculate_return(self, episode: Trajectory, offset: int, timestep: int) -> float:
        discount = jnp.cumprod(jnp.ones((offset + 1, )) * self.discount_factor)
        future_rewards = [r for _, _, r in episode[timestep - 1:]]  # note transition t points to `reward` t+1
        return (discount * future_rewards).sum()

    def _calculate_deltas(self, episode: Trajectory, offset: int) -> t.Tuple[t.FloatLike, t.FloatLike]:
        timestep = len(episode) - 1 - offset
        (state, action, _) = episode[timestep]
        G = self._calculate_return(episode, offset, timestep)
        delta = G - self.state_value(state.representation)

        value_grad = self.state_value.grad(state.representation)
        policy_grad = self.policy.gradlog(state.representation, action.representation)

        delta_value = self.learning_rate_value * delta * value_grad
        delta_policy = self.learning_rate_policy * self.discount_factor ** timestep * delta * policy_grad
        return delta_value, delta_policy

    def optimise(self, episode: Trajectory):
        for offset in range(len(episode)):
            delta_value, delta_policy = self._calculate_deltas(episode, offset)
            self.state_value.step(delta_value)
            self.policy.step(delta_policy)

    def batch_optimise(self,
                       episodes: t.Sequence[Trajectory],
                       mode: GradientAccumulationMode = GradientAccumulationMode.BATCH):
        delta_value = delta_policy = None
        n = 0
        for episode in episodes:
            for offset in range(len(episode)):
                dvalue, dpolicy = self._calculate_deltas(episode, offset)
                if delta_value is None:
                    delta_value, delta_policy = dvalue, dpolicy
                else:
                    delta_value += dvalue
                    delta_policy += dpolicy
                n += 1
            if mode == GradientAccumulationMode.EPISODE:
                self.state_value.step(delta_value / n)
                self.policy.step(delta_policy / n)
                delta_value *= 0
                delta_policy *= 0
                n = 0
        if mode == GradientAccumulationMode.BATCH:
            self.state_value.step(delta_value / n)
            self.policy.step(delta_policy / n)
