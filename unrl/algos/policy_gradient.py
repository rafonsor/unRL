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
from unrl.containers import FrozenTrajectory, Transition, Trajectory
from unrl.optim import optimiser_update, EligibilityTraceOptimizer
from unrl.utils import persisted_generator_value

DEFAULT_EPSILON = 0.1


class GradientAccumulationMode(Enum):
    STEP = 'step'
    EPISODE = 'episode'
    BATCH = 'batch'


class ActionSamplingMode(Enum):
    GREEDY = 'greedy'  # Always select the best-value actions, breaking ties randomly with equal probability
    EPSILON_GREEDY = 'epsilon'  # Select with probability `epsilon` one action uniformly at random, otherwise choose best-valued action
    STOCHASTIC = 'stochastic'  # Sample action according to distribution


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

    def __init__(self,
                 policy: Policy,
                 state_value_model: pt.nn.Module,
                 learning_rate_policy: float,
                 learning_rate_values: float,
                 discount_factor: float):
        self.policy = policy
        self.state_value_model = state_value_model
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate_policy
        self.learning_rate_values = learning_rate_values

    @persisted_generator_value
    def online_optimise(self,
                        starting_state: pt.Tensor,
                        mode: ActionSamplingMode = ActionSamplingMode.GREEDY,
                        **kwargs
                        ) -> t.Generator[t.IntLike, t.Tuple[t.IntLike, pt.Tensor, bool], Trajectory]:
        """Optimise policy and state-value function online for one episode.

        Args:
            starting_state:
                State from which to begin the episode.
            mode:
                strategy for sampling actions from the policy.
            **kwargs:
                Extra optional keyword arguments. For example, `epsilon` when using ActionSamplingMode.EPSILON_GREEDY.

        Yields:
            Action chosen by policy for the current state.

        Receives:
            Tuple (reward, successor, terminal) containing the reward obtained by applying the action yielded, the
            resulting successor state, and whether this state is terminal.

        Returns:
            A trajectory representing the full episode.

        """
        if mode == ActionSamplingMode.EPSILON_GREEDY:
            epsilon = kwargs.get('epsilon')
            if not epsilon:
                warnings.warn(RuntimeWarning(
                    f'EPSILON_GREEDY sampling chosen but no `epsilon` specific, using default value {DEFAULT_EPSILON}'))
                epsilon = DEFAULT_EPSILON

        episode = []
        state = starting_state
        I = 1
        terminal = False
        while not terminal:
            logprobs = self.policy(state)
            if mode == ActionSamplingMode.STOCHASTIC:
                action = pt.distributions.Categorical(logits=logprobs).sample()
            elif mode == ActionSamplingMode.EPSILON_GREEDY and pt.rand(1).item() <= epsilon:
                action = pt.randint(0, logprobs.shape[0], (1,))
            else:  # GREEDY and (1 - epsilon) case of EPSILON_GREEDY
                action = pt.argmax(logprobs)

            reward, next_state, terminal = yield action

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

    _optim_policy: EligibilityTraceOptimizer
    _optim_values: EligibilityTraceOptimizer

    def __init__(self,
                 policy: Policy,
                 state_value_model: pt.nn.Module,
                 discount_factor: float,
                 learning_rate_policy: float,
                 learning_rate_values: float,
                 trace_decay_policy: float,
                 trace_decay_values: float):
        self.policy = policy
        self.state_value_model = state_value_model
        self._optim_policy = EligibilityTraceOptimizer(
            policy.parameters(), discount_factor, learning_rate_policy, trace_decay_policy, discounted_gradient=True)
        self._optim_values = EligibilityTraceOptimizer(
            state_value_model.parameters(), discount_factor, learning_rate_values, trace_decay_values)
        self.discount_factor = discount_factor

    @persisted_generator_value
    def online_optimise(self,
                        starting_state: pt.Tensor,
                        mode: ActionSamplingMode = ActionSamplingMode.GREEDY,
                        **kwargs
                        ) -> t.Generator[t.IntLike, t.Tuple[t.IntLike, pt.Tensor, bool], Trajectory]:
        """Optimise policy and state-value function online for one episode.

        Args:
            starting_state:
                State from which to begin the episode.
            mode:
                strategy for sampling actions from the policy.
            **kwargs:
                Extra optional keyword arguments. For example, `epsilon` when using ActionSamplingMode.EPSILON_GREEDY.

        Yields:
            Action chosen by policy for the current state.

        Receives:
            Tuple (reward, successor, terminal) containing the reward obtained by applying the action yielded, the
            resulting successor state, and whether this state is terminal.

        Returns:
            A trajectory representing the full episode.

        """
        if mode == ActionSamplingMode.EPSILON_GREEDY:
            epsilon = kwargs.get('epsilon')
            if not epsilon:
                warnings.warn(RuntimeWarning(
                    f'EPSILON_GREEDY sampling chosen but no `epsilon` specific, using default value {DEFAULT_EPSILON}'))
                epsilon = DEFAULT_EPSILON

        episode = []
        state = starting_state
        terminal = False
        while not terminal:
            logprobs = self.policy(state)
            if mode == ActionSamplingMode.STOCHASTIC:
                action = pt.distributions.Categorical(logits=logprobs).sample()
            elif mode == ActionSamplingMode.EPSILON_GREEDY and pt.rand(1).item() <= epsilon:
                action = pt.randint(0, logprobs.shape[0], (1,))
            else:  # GREEDY and (1 - epsilon) case of EPSILON_GREEDY
                action = pt.argmax(logprobs)

            reward, next_state, terminal = yield action

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
