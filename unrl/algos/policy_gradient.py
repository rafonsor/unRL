#  Copyright 2023 The unRL Authors.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
from enum import Enum

import torch as pt

import unrl.types as t
from unrl.containers import FrozenTrajectory


class GradientAccumulationMode(Enum):
    STEP = 'step'
    EPISODE = 'episode'
    BATCH = 'batch'


class Policy(pt.nn.Module):
    pass


class Reinforce:
    """REINFORCE Monte-Carlo Policy-Gradient"""
    policy: Policy

    def __init__(self, policy: Policy, learning_rate: float, discount_factor: float):
        self.policy = policy
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate

    def optimise(self, episode: FrozenTrajectory):
        for offset in range(len(episode)):
            timestep = len(episode) - 1 - offset
            (state, action, _) = episode[timestep]

            # Compute and backpropagate gradients of log action probabilities relative to the chosen action
            logprobs = self.policy(state)
            logprobs[action].backward()

            G = self._calculate_return(episode, offset, timestep)
            step_and_magnitude = self.learning_rate * self.discount_factor ** timestep * G
            self._step(step_and_magnitude)

    def batch_optimise(self, episodes: t.Sequence[FrozenTrajectory]):
        for episode in episodes:
            self.optimise(episode)

    def _step(self, step_and_magnitude: t.FloatLike):
        """Step parameters in the direction of their gradients"""
        for p in self.policy.parameters():
            if p.requires_grad:
                assert p.grad, f"Attempting to perform gradient ascent on {p} with empty gradients"
                p += step_and_magnitude * p.grad
        self.policy.zero_grad()

    def _calculate_return(self, episode: FrozenTrajectory, offset: int, timestep: int) -> float:
        discount = pt.cumprod(pt.ones((offset + 1,)) * self.discount_factor, 0)
        future_rewards = episode.rewards[timestep - 1:]  # note transition t points to `reward` from time t+1
        return (discount * future_rewards).sum()
