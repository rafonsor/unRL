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
import gc
from copy import deepcopy

import torch as pt

import unrl.types as t
from unrl.algos.dqn import onestep_td_error
from unrl.config import validate_config
from unrl.containers import ContextualTransition, ContextualTrajectory
from unrl.experience_buffer import ExperienceBuffer
from unrl.functions import ContinuousPolicy, StateValueFunction, ContinuousActionValueFunction
from unrl.optim import multi_optimiser_stepper, polyak_averaging_inplace
from unrl.utils import persisted_generator_value


class DDPG:
    """Deep Deterministic Policy Gradient is a combination of principles from Policy Gradient and Q-learning approaches
    for deep learning models, specific for continuous action spaces in online settings. DDPG uses an Experience Replay
    buffer to be more sample efficient.

    References:
        [1] Lillicrap, T. P., Hunt, J. J., Pritzel, A. & et al. (2015). "Continuous control with deep reinforcement
            learning". arXiv:1509.02971.
    """
    policy: ContinuousPolicy
    target_policy: ContinuousPolicy
    action_value_model: ContinuousActionValueFunction
    target_action_value_model: ContinuousActionValueFunction

    def __init__(self,
                 policy: ContinuousPolicy,
                 action_value_model: ContinuousActionValueFunction,
                 discount_factor: float,
                 learning_rate_policy: float,
                 learning_rate_values: float,
                 noise_scale: float,
                 noise_exploration: float,
                 polyak_factor: float,
                 replay_memory_capacity: int,
                 batch_size: int,
                 target_refresh_steps: int):
        validate_config(discount_factor, 'discount_factor', 'unit')
        validate_config(learning_rate_policy, 'learning_rate_policy', 'unitpositive')
        validate_config(learning_rate_values, 'learning_rate_values', 'unitpositive')
        validate_config(noise_scale, 'noise_scale', 'positive')
        validate_config(noise_exploration, 'noise_exploration', 'unit')
        validate_config(polyak_factor, 'polyak_factor', 'unitpositive')
        validate_config(target_refresh_steps, 'target_refresh_steps', 'positive')
        validate_config(replay_memory_capacity, 'replay_memory_capacity', 'positive')
        validate_config(batch_size, 'batch_size', 'positive')
        self.policy = policy
        self.target_policy = deepcopy(policy)
        self.action_value_model = action_value_model
        self.target_action_value_model = deepcopy(action_value_model)
        self.discount_factor = discount_factor
        self.learning_rate_policy = learning_rate_policy
        self.learning_rate_values = learning_rate_values
        self.target_refresh_steps = target_refresh_steps
        self.polyak_weights = [polyak_factor, 1 - polyak_factor]

        self.batch_size = batch_size
        self._experience_buffer = ExperienceBuffer(maxlen=replay_memory_capacity)

        self._noise_process = pt.distributions.Normal(0, noise_scale)
        self.noise_exploration = noise_exploration

        self._optim_policy = pt.optim.SGD(self.policy.parameters(), lr=self.learning_rate_policy)
        self._optim_values = pt.optim.SGD(self.action_value_model.parameters(), lr=self.learning_rate_values)
        self._stepper = multi_optimiser_stepper(self._optim_policy, self._optim_values)
        self.__steps = 0

    def _store_and_sample(self, transition: ContextualTransition) -> t.Dict[str, pt.Tensor]:
        """Store the latest ContinuousSARST transition in the Experience buffer and sample a minibatch of past
        transitions.

        Args:
            transition: ContinuousSARST transition to add to the Experience buffer.

        Returns:
            A collection of stacked tensor representing a batch of ContinuousSARST transitions
        """
        self._experience_buffer.append(transition)
        batch, _ = self._experience_buffer.sample(self.batch_size)
        return batch

    def _compute_td_error_batch(self, batch: t.Dict[str, pt.Tensor]) -> pt.Tensor:
        """Compute One-step TD-errors for an entire batch of ContinuousSARST transitions.

        Args:
            batch: a collection of ContinuousSARST transitions sampled from an experience buffer. Each transition
                   element is batched together as a tensor.

        Returns:
            Unidimensional tensor of TD-errors
        """
        values = self.action_value_model(batch['states'], batch['actions'])
        successor_values = self.target_action_value_model(batch['next_states'], self.target_policy(batch['next_states']))
        return onestep_td_error(self.discount_factor, values, batch['rewards'], successor_values, batch['terminations'])

    @persisted_generator_value
    def online_optimise(
        self,
        starting_state: pt.Tensor
    ) -> t.Generator[t.IntLike, t.Tuple[t.IntLike, pt.Tensor, bool, bool], t.Tuple[ContextualTrajectory, float]]:
        """Optimise policy and state-value function online for one episode until termination (as determined by caller,
        can include e.g. a forced stoppage after a certain number of episodes).

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
            A tuple with a ContextualTrajectory of SARST transitions representing the full episode and the total loss
            of the experienced during the episode until termination.
        """
        episode = []
        total_loss = 0
        state = starting_state
        stop = False
        while not stop:
            self.__steps += 1
            action = self.policy(state)

            # Inject white noise
            noisy_action = action + self.noise_exploration * self._noise_process.sample(action.shape)
            action_value = self.action_value_model(state, noisy_action)

            reward, next_state, terminal, stop = yield noisy_action

            transition = ContextualTransition(state, noisy_action, reward, next_state, terminal)
            episode.append(transition)
            state = next_state

            batch = self._store_and_sample(transition)
            errors = self._compute_td_error_batch(batch)

            loss_values = (errors ** 2).mean()
            loss_policy = action_value
            # Note action-value function uses gradient descent whilst policy uses gradient ascent, it is thus subtracted
            loss = (loss_values - loss_policy)
            self._stepper(loss)
            total_loss += loss.item()

            if self.__steps % self.target_refresh_steps == 0:
                # update parameters of target models in proportion to the polyak factor
                polyak_averaging_inplace([self.target_policy, self.policy], self.polyak_weights)
                polyak_averaging_inplace([self.target_action_value_model, self.action_value_model], self.polyak_weights)

            del action_value
            del action
            del batch
            del errors

            if self.__steps % 250 == 0:
                gc.collect()

        return episode, total_loss / len(episode)


class TwinDelayedDDPG(DDPG):
    """Twin Delayed Deep Deterministic Policy Gradient is a DDPG variant based on clipped Double Q-learning.

    The "Delayed" qualifier comes from the fact that Policies are updated less frequently than Action-value functions
    (both behaviour and target). Whereas it is a "Twin" method because two parallel Action-value functions are used:
    during training, for every transition only the most conservative action-value estimates (resulting in lower absolute
    One-step TD-errors) are used to optimise both functions, thereby reducing an overestimation bias.

    References:
        [1] Fujimoto, S., Hoof, H., & Meger, D. (2018). "Addressing function approximation error in actor-critic
            methods". In Proceedings of the 35th International Conference on Machine Learning.
    """
    def __init__(self,
                 policy: ContinuousPolicy,
                 action_value_model: ContinuousActionValueFunction,
                 action_value_twin_model: ContinuousActionValueFunction,
                 discount_factor: float,
                 learning_rate_policy: float,
                 learning_rate_values: float,
                 noise_scale: float,
                 noise_exploration: float,
                 noise_epsilon: float,
                 polyak_factor: float,
                 replay_memory_capacity: int,
                 batch_size: int,
                 target_refresh_steps: int,
                 policy_update_delay: int):
        validate_config(noise_epsilon, 'noise_epsilon', 'positive')
        validate_config(policy_update_delay, 'policy_update_delay', 'positive')
        super().__init__(policy, action_value_model, discount_factor, learning_rate_policy, learning_rate_values,
                         noise_scale, noise_exploration, polyak_factor, replay_memory_capacity, batch_size,
                         target_refresh_steps)
        self.action_value_twin_model = action_value_twin_model
        self.target_action_value_twin_model = deepcopy(action_value_twin_model)
        self.noise_epsilon = noise_epsilon

        self._optim_values_twin = pt.optim.SGD(self.action_value_twin_model.parameters(), lr=self.learning_rate_values)
        self._stepper_policy = multi_optimiser_stepper(self._optim_policy)
        self._stepper_values = multi_optimiser_stepper(self._optim_values, self._optim_values_twin)
        self.__steps = 0
        self.policy_update_delay = policy_update_delay
        self.__update_policy = 0
        self.__update_target_policy = 0

    def _get_conservative_estimate(self, states: pt.Tensor, actions: pt.Tensor, use_target: bool = False) -> pt.Tensor:
        """Given state-action pairs, compute action-values and return the lowest among estimates from both functions"""
        assert states.shape[0] == actions.shape[0], "Inputs dimensions mismatch"
        if use_target:
            est1 = self.target_action_value_model(states, actions)
            est2 = self.target_action_value_twin_model(states, actions)
        else:
            est1 = self.action_value_model(states, actions)
            est2 = self.action_value_twin_model(states, actions)
        return pt.minimum(est1, est2)

    def _compute_td_error_batch(self, batch: t.Dict[str, pt.Tensor]) -> pt.Tensor:
        """Compute One-step TD-errors for an entire batch of ContinuousSARST transitions.

        Args:
            batch: a collection of ContinuousSARST transitions sampled from an experience buffer. Each transition
                   element is batched together as a tensor.

        Returns:
            Unidimensional tensor of TD-errors
        """
        # Get conservative estimates from known state-action pairs using behaviour Action-value functions
        values = self._get_conservative_estimate(batch['states'], batch['actions'])
        # To compute conservative estimates for next states, we add (bounded) noise to the actions of the target policy
        noise = pt.clip(self._noise_process.sample(batch['actions'].shape), -self.noise_epsilon, self.noise_epsilon)
        next_actions = self.target_policy(batch['next_states']) + noise
        successor_values = self._get_conservative_estimate(batch['next_states'], next_actions, use_target=True)
        return onestep_td_error(self.discount_factor, values, batch['rewards'], successor_values, batch['terminations'])

    @persisted_generator_value
    def online_optimise(
        self,
        starting_state: pt.Tensor
    ) -> t.Generator[t.IntLike, t.Tuple[t.IntLike, pt.Tensor, bool, bool], t.Tuple[ContextualTrajectory, float]]:
        """Optimise policy and state-value function online for one episode until termination (as determined by caller,
        can include e.g. a forced stoppage after a certain number of episodes).

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
            A tuple with a ContextualTrajectory of SARST transitions representing the full episode and the total loss
            of the experienced during the episode until termination.
        """
        episode = []
        total_loss = 0
        state = starting_state
        stop = False
        while not stop:
            self.__steps += 1
            action = self.policy(state)
            # Inject white noise
            noisy_action = action + self.noise_exploration * self._noise_process.sample(action.shape)

            reward, next_state, terminal, stop = yield noisy_action

            transition = ContextualTransition(state, noisy_action, reward, next_state, terminal)
            episode.append(transition)
            state = next_state

            batch = self._store_and_sample(transition)
            errors = self._compute_td_error_batch(batch)

            # Update Action-value functions
            loss_values = (errors ** 2).mean()
            self._stepper_values(loss_values)
            total_loss += loss_values.item()

            # Update Policy every `policy_update_delay` steps
            self.__update_policy = (self.__update_policy + 1) % self.policy_update_delay
            if self.__update_policy == 0:
                # Note: Policy loss uses only the first Action-state function
                loss_policy = self.action_value_model(batch['states'], self.policy(batch['states']))
                self._stepper_policy(loss_policy)
                total_loss += loss_policy.item()

            if self.__steps % self.target_refresh_steps == 0:
                self.__update_target_policy = (self.__update_target_policy + 1) % self.policy_update_delay
                # update parameters of target models in proportion to the polyak factor
                polyak_averaging_inplace([self.target_action_value_model, self.action_value_model], self.polyak_weights)
                polyak_averaging_inplace([self.target_action_value_twin_model, self.action_value_twin_model],
                                         self.polyak_weights)
                if self.__update_target_policy:
                    polyak_averaging_inplace([self.target_policy, self.policy], self.polyak_weights)

            del action
            del batch
            del errors

            if self.__steps % 250 == 0:
                gc.collect()

        return episode, total_loss / len(episode)
