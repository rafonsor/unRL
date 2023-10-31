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
import torch as pt

import unrl.types as t
from unrl.action_sampling import ActionSampler, make_sampler, ActionSamplingMode
from unrl.algos.dqn import onestep_td_error
from unrl.algos.policy_gradient import Policy
from unrl.containers import Transition, Trajectory
from unrl.optim import EligibilityTraceOptimizer
from unrl.utils import persisted_generator_value


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
        self._optim_policy = pt.optim.SGD(self.policy.parameters(), lr=self.learning_rate)
        self._optim_values = pt.optim.SGD(self.state_value_model.parameters(), lr=self.learning_rate_values)

    @persisted_generator_value
    def online_optimise(
        self,
        starting_state: pt.Tensor,
    ) -> t.Generator[t.IntLike, t.Tuple[t.IntLike, pt.Tensor, bool, bool], t.Tuple[Trajectory, float]]:
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
            A tuple (trajectory, average_loss) containing the trajectory of the full episode and the average TD-error.
        """
        episode = []
        total_loss = 0
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
            error = onestep_td_error(self.discount_factor, state_value, reward, next_state_value, terminal)

            # Compute and backpropagate gradients
            loss_policy = I * error * logprobs[action]
            loss_values = error * state_value
            loss = loss_values + loss_policy
            total_loss += loss.item()

            # Compute deltas and update parameters
            self._optim_policy.zero_grad()
            self._optim_values.zero_grad()
            loss.backward()
            self._optim_policy.step()
            self._optim_values.step()

            # Save transition
            episode.append(Transition(state, action.type(pt.int), reward, next_state))

            I *= self.discount_factor
            state = next_state

        return episode, total_loss / len(episode)


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
    def online_optimise(
        self,
        starting_state: pt.Tensor,
    ) -> t.Generator[t.IntLike, t.Tuple[t.IntLike, pt.Tensor, bool, bool], t.Tuple[Trajectory, float]]:
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
            A tuple (trajectory, average_loss) containing the trajectory of the full episode and the average TD-error.
        """
        episode = []
        total_loss = 0
        state = starting_state
        stop = False
        while not stop:
            logprobs = self.policy(state)
            action = self.sampler.sample(logprobs)

            reward, next_state, terminal, stop = yield action

            # Compute One-step TD-error
            state_value = self.state_value_model(state)
            next_state_value = pt.Tensor([0]) if terminal else self.state_value_model(next_state)
            error = onestep_td_error(self.discount_factor, state_value, reward, next_state_value, terminal)

            # Compute and backpropagate gradients
            loss_policy = error * logprobs[action]
            loss_values = error * state_value
            loss = loss_values + loss_policy
            total_loss += loss.item()

            loss.backward()
            self._optim_policy.step()
            self._optim_values.step()

            # Save transition
            episode.append(Transition(state, action.type(pt.int), reward, next_state))
            state = next_state

        self._optim_policy.episode_reset()
        self._optim_values.episode_reset()

        return episode, total_loss / len(episode)
