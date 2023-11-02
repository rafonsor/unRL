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
from copy import deepcopy
from enum import Enum

import torch as pt

import unrl.types as t
from unrl.action_sampling import make_sampler, ActionSamplingMode
from unrl.algos.dqn import onestep_td_error
from unrl.containers import FrozenTrajectory


class GradientAccumulationMode(Enum):
    STEP = 'step'
    EPISODE = 'episode'
    BATCH = 'batch'


class Policy(pt.nn.Module):
    def forward(self, state: pt.Tensor) -> pt.Tensor:
        """Returns logprobabilies of actions"""
        ...


class Reinforce:
    """REINFORCE Monte-Carlo Policy-Gradient"""
    policy: Policy

    def __init__(self, policy: Policy, learning_rate: float, discount_factor: float):
        self.policy = policy
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self._optim = pt.optim.SGD(self.policy.parameters(), lr=self.learning_rate)
        self._sampler = make_sampler(ActionSamplingMode.EPSILON_GREEDY)

    def sample(self, state: pt.Tensor) -> int:
        logprobs = self.policy(state)
        action = self._sampler.sample(logprobs)
        return action.item()

    def optimise(self, episode: FrozenTrajectory) -> float:
        total_loss = 0
        for offset in range(len(episode)):
            timestep = len(episode) - 1 - offset
            (state, action, _, _) = episode[timestep]
            G = self._calculate_return(episode, offset, timestep)
            logprobs = self.policy(state)

            loss = self.discount_factor ** timestep * G * logprobs[action]
            total_loss += loss.item()

            self._optim.zero_grad()
            loss.backward()
            self._optim.step()
        return total_loss / len(episode)

    def batch_optimise(self, episodes: t.Sequence[FrozenTrajectory]):
        for episode in episodes:
            self.optimise(episode)

    def _calculate_return(self, episode: FrozenTrajectory, offset: int, timestep: int) -> float:
        discount = pt.cumprod(pt.ones((offset + 1,)) * self.discount_factor, 0)
        future_rewards = episode.rewards[timestep:]  # note transition t points to `reward` from time t+1
        return (discount * future_rewards).sum()


class BaselineReinforce(Reinforce):
    """REINFORCE Monte-Carlo Policy-Gradient with state-value estimates as a return Baseline. Learns both a policy and a
     state-value function"""
    policy: Policy
    state_value_model: pt.nn.Module

    def __init__(self,
                 policy: Policy,
                 state_value_model: pt.nn.Module,
                 learning_rate_policy: float,
                 learning_rate_values: float,
                 discount_factor: float):
        super().__init__(policy, learning_rate_policy, discount_factor)
        self.state_value_model = state_value_model
        self.learning_rate_values = learning_rate_values
        self._optim_values = pt.optim.SGD(self.state_value_model.parameters(), lr=self.learning_rate_values)

    def optimise(self, episode: FrozenTrajectory) -> float:
        total_loss = 0
        for offset in range(len(episode)):
            timestep = len(episode) - 1 - offset
            (state, action, _, _) = episode[timestep]

            estimate = self.state_value_model(state)
            delta = self._calculate_return(episode, offset, timestep) - estimate
            logprobs = self.policy(state)

            loss_values = -delta * estimate
            loss_policy = -self.discount_factor ** timestep * delta * logprobs[action]
            loss = loss_values + loss_policy
            total_loss += loss.item()

            self._optim_values.zero_grad()
            self._optim.zero_grad()
            loss.backward()
            self._optim_values.step()
            self._optim.step()
        return total_loss / len(episode)


class SimplifiedPPO(Reinforce):
    """Simplified variant of Proximal Policy Optimisation ([1]_) based on pseudocode of [2]_ which does not make use of
    n-step returns.

    References:
        [1] Schulman, J., Wolski, F., Dhariwal, P., & et al. (2017). "Proximal policy optimization algorithms".
            arXiv:1707.06347.
        [2] Albrecht S. V., Christianos F. & Schäfer L. (2024). Section 8.2.6., Multi-Agent Reinforcement Learning:
            Foundations and Modern Approaches. Pre-print.
    """
    policy: Policy
    behaviour_policy: Policy
    state_value_model: pt.nn.Module

    def __init__(self,
                 policy: Policy,
                 state_value_model: pt.nn.Module,
                 learning_rate_policy: float,
                 learning_rate_values: float,
                 discount_factor: float,
                 epsilon: float,
                 target_refresh_steps: int):
        super().__init__(policy, learning_rate_policy, discount_factor)
        self.state_value_model = state_value_model
        self.learning_rate_values = learning_rate_values
        self.epsilon = epsilon
        self.target_refresh_steps = target_refresh_steps
        self._optim_values = pt.optim.SGD(self.state_value_model.parameters(), lr=self.learning_rate_values)
        self.target_policy = deepcopy(self.policy)
        self.__steps = 0

    def optimise(self, episode: FrozenTrajectory) -> float:
        total_loss = 0
        for offset in range(len(episode)):
            timestep = len(episode) - 1 - offset
            (state, action, reward, next_state, terminal) = episode[timestep]

            estimate = self.state_value_model(state)
            next_state_estimate = self.state_value_model(next_state)
            logprobs = self.target_policy(state)
            logprobs_b = self.policy(state)

            advantage = onestep_td_error(self.discount_factor, estimate, reward, next_state_estimate, terminal)
            rho = logprobs[action] / logprobs_b[action]

            loss_values = advantage ** 2
            # Note: minimisation coupled with clipping ensures lower bounding of loss when advantage is negative
            loss_policy = -min(rho * advantage, pt.clip(rho, 1 - self.epsilon, 1 + self.epsilon) * advantage)
            loss = loss_values + loss_policy
            total_loss += loss.item()

            self._optim_values.zero_grad()
            self._optim.zero_grad()
            loss.backward()
            self._optim_values.step()
            self._optim.step()

            self.__steps += 1
            if self.__steps % self.target_refresh_steps == 0:
                self.target_policy.load_state_dict(self.policy.state_dict())
                self.__steps = 0
        return total_loss / len(episode)
