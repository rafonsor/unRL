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
import warnings
from enum import Enum

import torch as pt

import unrl.types as t
from unrl.action_sampling import ActionSampler, make_sampler, ActionSamplingMode
from unrl.containers import FrozenTrajectory, Transition, Trajectory
from unrl.optim import optimiser_update, EligibilityTraceOptimizer
from unrl.utils import persisted_generator_value


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

    def optimise(self, episode: FrozenTrajectory):
        for offset in range(len(episode)):
            timestep = len(episode) - 1 - offset
            (state, action, _) = episode[timestep]

            # Compute and backpropagate gradients of log action probabilities relative to the chosen action
            logprobs = self.policy(state)
            logprobs[action].backward()
            # if mode == GradientAccumulationMode.STEP:
            #     logprobs[action].backward()
            # else:
            #     grad = logprobs[action].grad_fn(logprobs[action])

            G = self._calculate_return(episode, offset, timestep)
            step_and_magnitude = self.learning_rate * self.discount_factor ** timestep * G
            optimiser_update(self.policy, step_and_magnitude)

    def batch_optimise(self, episodes: t.Sequence[FrozenTrajectory]):
        for episode in episodes:
            self.optimise(episode)

    def _calculate_return(self, episode: FrozenTrajectory, offset: int, timestep: int) -> float:
        discount = pt.cumprod(pt.ones((offset + 1,)) * self.discount_factor, 0)
        future_rewards = episode.rewards[timestep - 1:]  # note transition t points to `reward` from time t+1
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

    def optimise(self, episode: FrozenTrajectory):
        for offset in range(len(episode)):
            timestep = len(episode) - 1 - offset
            (state, action, _) = episode[timestep]

            # Compute and backpropagate gradients
            logprobs = self.policy(state)
            logprobs[action].backward()
            estimate = self.state_value_model(state)
            estimate.backward()

            # Compute deltas and update parameters
            delta = self._calculate_return(episode, offset, timestep) - estimate
            step_and_magnitude_values = self.learning_rate_values * delta
            step_and_magnitude_policy = self.learning_rate * self.discount_factor ** timestep * delta
            optimiser_update(self.state_value_model, step_and_magnitude_values)
            optimiser_update(self.policy, step_and_magnitude_policy)


class ActorCritic:
    """One-Step Online Actor-Critic for episodic settings"""
    policy: Policy
    state_value_model: pt.nn.Module
    sampler: ActionSampler  # Strategy for sampling actions from the policy

    def __init__(self,
                 policy: Policy,
                 state_value_model: pt.nn.Module,
                 learning_rate_policy: float,
                 learning_rate_values: float,
                 discount_factor: float,
                 action_sampler: t.Optional[ActionSampler] = None):
        """Args:
            policy: Policy model to improve
            state_value_model: State-value model to improve
            learning_rate_policy: learning rate for Policy model
            learning_rate_values: learning rate for State-value mode
            discount_factor: discounting rate for future rewards
            action_sampler: (Optional) strategy for sampling actions from the policy. Defaults to Greedy sampling if not
                            provided.
        """
        self.policy = policy
        self.state_value_model = state_value_model
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate_policy
        self.learning_rate_values = learning_rate_values
        self.sampler = action_sampler or make_sampler(ActionSamplingMode.GREEDY)

    @persisted_generator_value
    def online_optimise(self,
                        starting_state: pt.Tensor,
                        ) -> t.Generator[t.IntLike, t.Tuple[t.IntLike, pt.Tensor, bool, bool], Trajectory]:
        """Optimise policy and state-value function online for one episode.

        Args:
            starting_state:
                State from which to begin the episode.

        Yields:
            Action chosen by policy for the current state.

        Receives:
            Tuple (reward, successor, terminal, stop) containing the reward obtained by applying the action yielded, the
            resulting successor state, and whether this state is terminal. And, finally, an indication of whether to
            stop irrespective of the ending up in a terminal state.

        Returns:
            A trajectory representing the full episode.

        """
        episode = []
        state = starting_state
        I = 1
        stop = False
        while not stop:
            logprobs = self.policy(state)
            action = self.sampler.sample(logprobs)

            reward, next_state, terminal, stop = yield action

            # Compute One-step TD-error
            state_value = self.state_value_model(state)
            next_state_value = pt.Tensor([0]) if terminal else self.state_value_model(next_state)
            error = reward + self.discount_factor * next_state_value - state_value

            # Compute and backpropagate gradients
            logprobs[action].backward()
            state_value.backward()

            # Compute deltas and update parameters
            step_and_magnitude_values = self.learning_rate_values * error
            step_and_magnitude_policy = self.learning_rate * I * error
            optimiser_update(self.state_value_model, step_and_magnitude_values)
            optimiser_update(self.policy, step_and_magnitude_policy)

            # Save transition
            episode.append(Transition(state, action.type(pt.int), reward, next_state))

            I *= self.discount_factor
            state = next_state

        return episode


class EligibilityTraceActorCritic:
    """One-Step Online Actor-Critic with eligibility traces for episodic settings"""
    policy: Policy
    state_value_model: pt.nn.Module
    sampler: ActionSampler

    _optim_policy: EligibilityTraceOptimizer
    _optim_values: EligibilityTraceOptimizer

    def __init__(self,
                 policy: Policy,
                 state_value_model: pt.nn.Module,
                 *,
                 discount_factor: float,
                 learning_rate_policy: float,
                 learning_rate_values: float,
                 trace_decay_policy: float,
                 trace_decay_values: float,
                 weight_decay_policy: float = 0.0,
                 weight_decay_values: float = 0.0,
                 action_sampler: t.Optional[ActionSampler] = None):
        """Args:
            policy: Policy model to improve
            state_value_model: State-value model to improve
            discount_factor: discounting rate for future rewards
            learning_rate_policy: learning rate for Policy model
            learning_rate_values: learning rate for State-value mode
            trace_decay_policy: decay factor for eligibility traces of Policy model
            trace_decay_values: decay factor for eligibility traces of State-value mode
            weight_decay_policy: (Optional) weight decay rates for Policy model. Disabled if set to "0.0".
            weight_decay_values: (Optional) weight decay rates for State-value mode. Disabled if set to "0.0".
            action_sampler: (Optional) strategy for sampling actions from the policy. Defaults to Greedy sampling if not
                            provided.
        """
        self.policy = policy
        self.state_value_model = state_value_model
        self.discount_factor = discount_factor
        self._optim_policy = EligibilityTraceOptimizer(
            policy.parameters(), discount_factor, learning_rate_policy, trace_decay_policy,
            weight_decay=weight_decay_policy, discounted_gradient=True)
        self._optim_values = EligibilityTraceOptimizer(
            state_value_model.parameters(), discount_factor, learning_rate_values, trace_decay_values,
            weight_decay=weight_decay_values)
        self.sampler = action_sampler or make_sampler(ActionSamplingMode.GREEDY)

    @persisted_generator_value
    def online_optimise(self,
                        starting_state: pt.Tensor,
                        ) -> t.Generator[t.IntLike, t.Tuple[t.IntLike, pt.Tensor, bool, bool], Trajectory]:
        """Optimise policy and state-value function online for one episode.

        Args:
            starting_state:
                State from which to begin the episode.

        Yields:
            Action chosen by policy for the current state.

        Receives:
            Tuple (reward, successor, terminal, bool) containing the reward obtained by applying the action yielded, the
            resulting successor state, and whether this state is terminal. And, finally, an indication of whether to
            stop irrespective of the ending up in a terminal state.

        Returns:
            A trajectory representing the full episode.
        """
        episode = []
        state = starting_state
        stop = False
        while not stop:
            logprobs = self.policy(state)
            action = self.sampler.sample(logprobs)

            reward, next_state, terminal, stop = yield action

            # Compute One-step TD-error
            state_value = self.state_value_model(state)
            next_state_value = pt.Tensor([0]) if terminal else self.state_value_model(next_state)
            error = reward + self.discount_factor * next_state_value - state_value

            # Compute and backpropagate gradients
            logprobs[action].backward()
            state_value.backward()

            # Update parameters
            self._optim_policy.step(lambda: error.item())
            self._optim_values.step(lambda: error.item())

            # Save transition
            episode.append(Transition(state, action.type(pt.int), reward, next_state))
            state = next_state

        self._optim_policy.episode_reset()
        self._optim_values.episode_reset()

        return episode
