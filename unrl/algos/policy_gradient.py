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
from unrl.basic import entropy
from unrl.config import validate_config
from unrl.containers import FrozenTrajectory
from unrl.utils import multi_optimiser_stepper


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


class TRPO(Reinforce):
    """Trust Region Policy Optimisation (TRPO) proposed by [1]_ is a policy gradient algorithm that constrains the
    magnitude of updates by subjecting them to a penalty based on the KL divergence between behaviour and target
    policies.

    This penalty is controlled by a fixed positive coefficient `beta`. A heuristic for an adaptive coefficient is
    evaluated in [2]_. To enable this heuristic, `kl_divergence_reference`, which defines a reference divergence to
    compared against to alter `beta`, must be set to a positive value.

    References:
        [1] Schulman, J., Levine, S., Abbeel, & et al. (2015). "Trust region policy optimization". In Proceedings of The
            32nd International Conference on Machine Learning.
        [2] Schulman, J., Wolski, F., Dhariwal, P., & et al. (2017). "Proximal policy optimization algorithms".
            arXiv:1707.06347.
    """
    policy: Policy
    target_policy: Policy
    state_value_model: pt.nn.Module

    def __init__(self,
                 policy: Policy,
                 state_value_model: pt.nn.Module,
                 learning_rate_policy: float,
                 learning_rate_values: float,
                 discount_factor: float,
                 lambda_factor: float,
                 beta: float,
                 target_refresh_steps: int,
                 kl_divergence_reference: t.Optional[float] = None):
        validate_config(lambda_factor, 'lambda_factor', 'unit')
        validate_config(beta, 'beta', 'positive')
        validate_config(target_refresh_steps, 'target_refresh_steps', 'positive')
        validate_config(learning_rate_values, 'learning_rate_values', 'unitpositive')
        if kl_divergence_reference:
            validate_config(kl_divergence_reference, 'kl_divergence_reference', 'positive')

        super().__init__(policy, learning_rate_policy, discount_factor)
        self.state_value_model = state_value_model
        self.learning_rate_values = learning_rate_values
        self.lam = lambda_factor
        self.target_refresh_steps = target_refresh_steps
        self.kl_divergence_reference = kl_divergence_reference

        self._optim_values = pt.optim.SGD(self.state_value_model.parameters(), lr=self.learning_rate_values)
        self._stepper = multi_optimiser_stepper(self._optim, self._optim_values)
        self.target_policy = deepcopy(self.policy)
        self.__steps = 0
        self.__beta = beta

    @property
    def beta(self) -> float:
        return self.__beta

    def optimise(self, episode: FrozenTrajectory) -> float:
        # Compute One-step Advantages for all transitions, these are later composed with variable discounting.
        advantages = []
        for transition in episode:
            (state, _, reward, next_state, terminal) = transition
            estimate = self.state_value_model(state)
            next_state_estimate = self.state_value_model(next_state)
            advantages.append(onestep_td_error(self.discount_factor, estimate, reward, next_state_estimate, terminal))

        advantages = pt.Stack(advantages)
        discounts = pt.cumprod(pt.arange(0, len(episode)) * self.lam * self.discount_factor, dim=-1)
        offset = 0

        # Compute and aggregate surrogate losses
        total_loss = 0
        for timestep, transition in enumerate(episode):
            (state, action, _, _, _) = transition
            logprobs = self.target_policy(state)
            logprobs_b = self.policy(state)

            # Importance sampling weight
            rho = logprobs[action] / logprobs_b[action]

            # n-step Advantage estimate
            nae = advantages[timestep:] * discounts[:-offset]
            offset += 1

            kl_divergence = pt.kl_div(logprobs_b, logprobs, log_target=True)
            loss_policy = rho * nae - self.__beta * kl_divergence
            loss_values = advantages[timestep] ** 2
            total_loss += loss_policy + loss_values

            if self.kl_divergence_reference is not None:
                # Adapt KL penalty coefficient according to heuristic proposed by [2]_
                if kl_divergence > self.kl_divergence_reference * 1.5:
                    self.__beta *= 2
                elif kl_divergence < self.kl_divergence_reference / 1.5:
                    self.__beta /= 2

        self._stepper(total_loss)

        self.__steps += 1
        if self.__steps % self.target_refresh_steps == 0:
            self.target_policy.load_state_dict(self.policy.state_dict())
            self.__steps = 0
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
        validate_config(epsilon, 'epsilon', 'positive')
        validate_config(target_refresh_steps, 'target_refresh_steps', 'positive')
        validate_config(learning_rate_values, 'learning_rate_values', 'unitpositive')
        super().__init__(policy, learning_rate_policy, discount_factor)
        self.state_value_model = state_value_model
        self.learning_rate_values = learning_rate_values
        self.epsilon = epsilon
        self.target_refresh_steps = target_refresh_steps
        self.target_policy = deepcopy(self.policy)
        self._optim_values = pt.optim.SGD(self.state_value_model.parameters(), lr=self.learning_rate_values)
        self._stepper = multi_optimiser_stepper(self._optim, self._optim_values)
        self.__steps = 0

    def optimise(self, episode: FrozenTrajectory) -> float:
        total_loss = 0
        for transition in episode:
            (state, action, reward, next_state, terminal) = transition

            estimate = self.state_value_model(state)
            next_state_estimate = self.state_value_model(next_state)
            logprobs = self.target_policy(state)
            logprobs_b = self.policy(state)

            advantage = onestep_td_error(self.discount_factor, estimate, reward, next_state_estimate, terminal)
            rho = logprobs[action] / logprobs_b[action]

            # Note: minimisation coupled with clipping ensures loss is lower bounded when Advantage is negative
            loss_policy = -min(rho * advantage, pt.clip(rho, 1 - self.epsilon, 1 + self.epsilon) * advantage)
            loss_values = advantage ** 2
            loss = loss_policy - loss_values
            total_loss += loss.item()

        self._stepper(total_loss)

        self.__steps += 1
        if self.__steps % self.target_refresh_steps == 0:
            self.target_policy.load_state_dict(self.policy.state_dict())
            self.__steps = 0
        return total_loss / len(episode)


class PPO(Reinforce):
    """Proximal Policy Optimisation (PPO) algorithm proposed by [1]_. Policy loss is constructed from unbiased n-step
    Advantage estimates and clipped within a bounded region determined by `epsilon`. To debias estimates PPO relies on a
    dual behaviour-target policies like Actor-Critic methods.

    References:
        [1] Schulman, J., Wolski, F., Dhariwal, P., & et al. (2017). "Proximal policy optimization algorithms".
            arXiv:1707.06347.
    """
    policy: Policy
    target_policy: Policy
    state_value_model: pt.nn.Module

    def __init__(self,
                 policy: Policy,
                 state_value_model: pt.nn.Module,
                 learning_rate_policy: float,
                 learning_rate_values: float,
                 discount_factor: float,
                 lambda_factor: float,
                 epsilon: float,
                 target_refresh_steps: int,
                 value_loss_coefficient: float = 1.0,
                 entropy_loss_coefficient: float = 0.0):
        validate_config(value_loss_coefficient, 'value_loss_coefficient', 'unit')
        validate_config(entropy_loss_coefficient, 'entropy_loss_coefficient', 'unit')
        validate_config(lambda_factor, 'lambda_factor', 'unit')
        validate_config(epsilon, 'epsilon', 'positive')
        validate_config(target_refresh_steps, 'target_refresh_steps', 'positive')
        validate_config(learning_rate_values, 'learning_rate_values', 'unitpositive')
        super().__init__(policy, learning_rate_policy, discount_factor)
        self.state_value_model = state_value_model
        self.learning_rate_values = learning_rate_values
        self.lam = lambda_factor
        self.epsilon = epsilon
        self.target_refresh_steps = target_refresh_steps
        self.value_loss_coefficient = value_loss_coefficient
        self.entropy_loss_coefficient = entropy_loss_coefficient

        self._optim_values = pt.optim.SGD(self.state_value_model.parameters(), lr=self.learning_rate_values)
        self._stepper = multi_optimiser_stepper(self._optim, self._optim_values)
        self.target_policy = deepcopy(self.policy)
        self.__steps = 0

    def optimise(self, episode: FrozenTrajectory) -> float:
        # Compute One-step Advantages for all transitions, these are later composed with variable discounting.
        advantages = []
        for transition in episode:
            (state, _, reward, next_state, terminal) = transition
            estimate = self.state_value_model(state)
            next_state_estimate = self.state_value_model(next_state)
            advantages.append(onestep_td_error(self.discount_factor, estimate, reward, next_state_estimate, terminal))

        advantages = pt.Stack(advantages)
        discounts = pt.cumprod(pt.arange(0, len(episode)) * self.lam * self.discount_factor, dim=-1)
        offset = 0

        # Compute and aggregate surrogate losses
        total_loss = 0
        for timestep, transition in enumerate(episode):
            (state, action, _, _, _) = transition
            logprobs = self.target_policy(state)
            logprobs_b = self.policy(state)

            # Importance sampling weight
            rho = logprobs[action] / logprobs_b[action]

            # n-step Advantage estimate
            nae = advantages[timestep:] * discounts[:-offset]
            offset += 1

            # Note: minimisation is coupled with clipping to ensure policy loss is lower bounded when Advantage is
            # negative (see fig.1 [1]_).
            loss_policy = min(rho * nae, pt.clip(rho, 1 - self.epsilon, 1 + self.epsilon) * nae)
            loss_values = self.value_loss_coefficient * (advantages[timestep] ** 2)
            loss_entropy = self.entropy_loss_coefficient * entropy(logits=logprobs)
            # Note: In [1]_ eq.9 the state-value function loss is subtracted only because authors decided to invert the
            # signs of the one-step Advantage equation.
            total_loss += loss_policy + loss_values + loss_entropy

        self._stepper(total_loss)

        self.__steps += 1
        if self.__steps % self.target_refresh_steps == 0:
            self.target_policy.load_state_dict(self.policy.state_dict())
            self.__steps = 0
        return total_loss / len(episode)
