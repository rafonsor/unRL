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
            self._step(self.policy, step_and_magnitude)

    def batch_optimise(self, episodes: t.Sequence[FrozenTrajectory]):
        for episode in episodes:
            self.optimise(episode)

    @staticmethod
    def _step(model: pt.nn.Module, step_and_magnitude: t.FloatLike):
        """Step parameters in the direction of their gradients"""
        for p in model.parameters():
            if p.requires_grad:
                assert p.grad, f"Attempting to perform gradient ascent on {p} with empty gradients"
                p += step_and_magnitude * p.grad
        model.zero_grad()

    def _calculate_return(self, episode: FrozenTrajectory, offset: int, timestep: int) -> float:
        discount = pt.cumprod(pt.ones((offset + 1,)) * self.discount_factor, 0)
        future_rewards = episode.rewards[timestep - 1:]  # note transition t points to `reward` from time t+1
        return (discount * future_rewards).sum()


class BaselineReinforce(Reinforce):
    """REINFORCE Monte-Carlo Policy-Gradient with Baseline that learns both a policy and state-values"""
    policy: Policy
    state_values: pt.nn.Module

    def __init__(self,
                 policy: Policy,
                 state_values: pt.nn.Module,
                 learning_rate_policy: float,
                 learning_rate_values: float,
                 discount_factor: float):
        super().__init__(policy, learning_rate_policy, discount_factor)
        self.state_values = state_values
        self.learning_rate_values = learning_rate_values

    def optimise(self, episode: FrozenTrajectory):
        for offset in range(len(episode)):
            timestep = len(episode) - 1 - offset
            (state, action, _) = episode[timestep]

            # Compute and backpropagate gradients
            logprobs = self.policy(state)
            logprobs[action].backward()
            estimate = self.state_values(state).sum()
            estimate.backward()

            # Compute deltas and update parameters
            delta = self._calculate_return(episode, offset, timestep) - estimate
            step_and_magnitude_values = self.learning_rate_values * delta
            step_and_magnitude_policy = self.learning_rate * self.discount_factor ** timestep * delta
            self._step(self.state_values, step_and_magnitude_values)
            self._step(self.policy, step_and_magnitude_policy)
