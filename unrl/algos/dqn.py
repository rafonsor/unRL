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
"""Deep Q-Learning and variants"""
import gc
from abc import ABCMeta, abstractmethod
from copy import deepcopy

import torch as pt
from torch.optim import SGD

import unrl.types as t
from unrl.config import validate_config
from unrl.action_sampling import EpsilonGreedyActionSampler, GreedyActionSampler
from unrl.containers import ContextualTransition, ContextualTrajectory
from unrl.experience_buffer import ExperienceBufferProtocol, ExperienceBuffer, NaivePrioritisedExperienceBuffer, \
    RankPartitionedPrioritisedExperienceBuffer
from unrl.functions import ActionValueFunction
from unrl.utils import persisted_generator_value

__all__ = [
    "DQN",
    "DQNExperienceReplay",
    "DQNPrioritisedExperienceReplay",
    "DoubleDQN",
    "PrioritisedDoubleDQN"
]


def onestep_td_error(discount_factor: t.FloatLike,
                     value: pt.Tensor,
                     reward: t.FloatLike,
                     successor_value: pt.Tensor,
                     terminal: t.BoolLike,
                     ) -> pt.Tensor:
    """Computes One-step TD-error ``r + γQ(s_{t+1},a_{t+1}) - Q(s_t, a_t)`` for a single transition or a batch.

    Args:
        reward:
        discount_factor:
        value:
        successor_value:
        terminal:

    Returns:
        Unidimensional tensor of One-step TD-error depending on the inputs size.
    """
    return reward + (1 - terminal) * discount_factor * successor_value - value


class DQN:
    """Basic Deep Q-Network (without an Experience Replay buffer), adapted from [1]_.

    References:
        [1] Mnih, V., Kavukcuoglu, K. & Silver, D., et al. (2015). "Human-level control through deep reinforcement
            learning". Nature, 518, pp. 529-533.
    """
    def __init__(self,
                 action_value_model: ActionValueFunction,
                 *,
                 learning_rate: float,
                 discount_factor: float,
                 epsilon_greedy: float,
                 target_refresh_steps: int):
        validate_config(learning_rate, "learning_rate", "unitpositive")
        validate_config(discount_factor, "discount_factor", "unitpositive")
        validate_config(target_refresh_steps, "target_refresh_steps", "positive")
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.target_refresh_steps = target_refresh_steps

        self.behaviour_model = action_value_model
        self.target_model = deepcopy(self.behaviour_model)
        self._sampler = EpsilonGreedyActionSampler(epsilon_greedy)
        self._optim = SGD(self.behaviour_model.parameters(), lr=learning_rate)

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
        step = 0
        while not stop:
            step += 1
            action_values = self.behaviour_model(state)
            action = self._sampler.sample(action_values)

            reward, next_state, terminal, stop = yield action

            transition = ContextualTransition(state, action.type(pt.int), reward, next_state, terminal)
            episode.append(transition)
            state = next_state

            error = self._process_transition(action_values, transition)
            total_loss += self._step(error)

            if step % self.target_refresh_steps == 0:
                self.target_model.load_state_dict(self.behaviour_model.state_dict())

            del action_values
            del action
            del error

            if step % 1000 == 0:
                gc.collect()

        return episode, total_loss / len(episode)

    def _process_transition(self, action_values: pt.Tensor, transition: ContextualTransition) -> pt.Tensor:
        """Compute the One-step TD-error using the SARST transition and the current state's action-value estimates
        provided by the behaviour model."""
        return self._compute_td_error_transition(action_values, transition)

    def _compute_td_error_transition(self, action_values: pt.Tensor, transition: ContextualTransition) -> pt.Tensor:
        return onestep_td_error(
            self.discount_factor,
            action_values[transition.action],
            transition.reward,
            self.target_model(transition.next_state).max(),
            transition.terminates)

    def _step(self, td: t.FloatLike) -> float:
        """Compute MSE loss from One-step TD-error to backpropagate gradients and update the behaviour model."""
        loss = self._compute_loss(td)
        loss.backward()
        self._optim.step()
        self._optim.zero_grad()
        return loss.item()

    @staticmethod
    def _compute_loss(td: t.FloatLike) -> pt.Tensor:
        """Compute MSE loss relative to One-step TD-error"""
        return (td ** 2).mean()


class _DQNExperienceReplayBase(DQN, metaclass=ABCMeta):
    """Base implementation of a Deep Q-Network with support for experience replay. Must be specialised with a concrete
    implementation of an Experience buffer matching ExperienceBufferProtocol."""

    def __init__(self,
                 action_value_model: ActionValueFunction,
                 *,
                 learning_rate: float,
                 discount_factor: float,
                 epsilon_greedy: float,
                 target_refresh_steps: int,
                 replay_memory_capacity: int,
                 batch_size: int):
        super().__init__(action_value_model, learning_rate=learning_rate, discount_factor=discount_factor,
                         epsilon_greedy=epsilon_greedy, target_refresh_steps=target_refresh_steps)
        self.replay_memory_capacity = replay_memory_capacity
        self.batch_size = batch_size

    def _process_transition(self, action_values: pt.Tensor, transition: ContextualTransition) -> pt.Tensor:
        """Compute the One-step TD-errors for a minibatch of sampled experiences. The experienced SARST transition is
        added to the Experience buffer beforehand.

        Action-value estimates for all starting states are recomputed on the spot using the behaviour model.

        Args:
            action_values: (NotUsed)
            transition: SARST transition to add to the Experience buffer.

        Returns:
            One-step TD-errors for each sampled transition as a unidimensional tensor
        """
        self._store_transition(action_values, transition)
        batch, _ = self.experience_buffer.sample(self.batch_size)
        return self._compute_td_error_batch(batch)

    def _compute_td_error_batch(self, batch: t.Dict[str, pt.Tensor]) -> pt.Tensor:
        """Compute One-step TD-errors for an entire batch of SARST transitions.

        Args:
            batch: a collection of SARST transitions sampled from an experience buffer. Each transition element is
                   batched together as a tensor.

        Returns:
            Unidimensional tensor of TD-errors
        """
        values = pt.take_along_dim(self.behaviour_model(batch['states']), batch['actions'].type(pt.long), dim=-1)[:, 0]
        successor_values = self.target_model(batch['next_states']).max(dim=-1).values
        return onestep_td_error(self.discount_factor, values, batch['rewards'], successor_values, batch['terminations'])

    @abstractmethod
    def _store_transition(self, action_values: pt.Tensor, transition: ContextualTransition):
        """Add transition to experience buffer for latter sampling and replay

        Args:
            action_values: action-value estimates for the transition's starting state.
            transition: a transition of the SARST form to store in the experience buffer.
        """
        ...

    @property
    @abstractmethod
    def experience_buffer(self) -> ExperienceBufferProtocol:
        """Experience buffer instance"""
        ...


class DQNExperienceReplay(_DQNExperienceReplayBase):
    """Deep Q-Network with an Experience Replay buffer as originally proposed by [1]_.

    References:
        [1] Mnih, V., Kavukcuoglu, K. & Silver, D., et al. (2015). "Human-level control through deep reinforcement
            learning". Nature, 518, pp. 529-533.
    """
    def __init__(self,
                 action_value_model: ActionValueFunction,
                 *,
                 learning_rate: float,
                 discount_factor: float,
                 epsilon_greedy: float,
                 target_refresh_steps: int,
                 replay_memory_capacity: int,
                 batch_size: int):
        super().__init__(
            action_value_model, learning_rate=learning_rate, discount_factor=discount_factor,
            epsilon_greedy=epsilon_greedy, target_refresh_steps=target_refresh_steps, batch_size=batch_size,
            replay_memory_capacity=replay_memory_capacity)
        self._experience_buffer = ExperienceBuffer(maxlen=self.replay_memory_capacity)

    def _store_transition(self, action_values: pt.Tensor, transition: ContextualTransition):
        self.experience_buffer.append(transition)

    @property
    def experience_buffer(self) -> ExperienceBufferProtocol:
        return self._experience_buffer


class DQNPrioritisedExperienceReplay(_DQNExperienceReplayBase):
    """Deep Q-Network with a prioritised Experience Replay buffer that samples transitions in proportion to their
    absolute One-step TD-error. Note this does not include double Q-learning as proposed by [1]_, use
    "PrioritisedDoubleDQN".

    References:
        [1] Schaul, T., Quan, J., Antonoglou, I., & et al. (2015). "Prioritized experience replay". arXiv:1511.05952.
    """

    alpha: float
    beta: float

    def __init__(self,
                 action_value_model: ActionValueFunction,
                 *,
                 learning_rate: float,
                 discount_factor: float,
                 epsilon_greedy: float,
                 target_refresh_steps: int,
                 replay_memory_capacity: int,
                 batch_size: int,
                 alpha: float,
                 beta: float,
                 use_ranks: bool = True):
        super().__init__(action_value_model, learning_rate=learning_rate, discount_factor=discount_factor,
                         epsilon_greedy=epsilon_greedy, target_refresh_steps=target_refresh_steps,
                         replay_memory_capacity=replay_memory_capacity, batch_size=batch_size)
        if use_ranks:
            self._experience_buffer = RankPartitionedPrioritisedExperienceBuffer(maxlen=self.replay_memory_capacity,
                                                                                 partitions=batch_size)
        else:
            self._experience_buffer = NaivePrioritisedExperienceBuffer(maxlen=self.replay_memory_capacity)
        self.alpha = alpha
        self.beta = beta

    @property
    def experience_buffer(self) -> ExperienceBufferProtocol:
        return self._experience_buffer

    def _store_transition(self, action_values: pt.Tensor, transition: ContextualTransition):
        priority = self._compute_priority(action_values, transition)
        self.experience_buffer.append(transition, priority)

    def _compute_priority(self, action_values: pt.Tensor, transition: ContextualTransition) -> pt.Tensor:
        """Assign a priority value to a transition using the action-value estimates computed for the starting state.
        [1]_ proposes to use the absolute One-step TD-error to assign a priority value. For sampling, priorities are
        exponentiated by the coefficient `alpha`. This operation is performed beforehand to remove implementation
        details from the prioritised experience buffer used.

        Args:
            action_values: action-value estimates from the behaviour model for the starting state.
            transition: experienced transition of the SARST form.

        Returns:
            A single priority value, exponentiated by `alpha`.
        """
        return self._compute_td_error_transition(action_values, transition).abs() ** self.alpha

    def _process_transition(self, action_values: pt.Tensor, transition: ContextualTransition) -> pt.Tensor:
        """Compute the One-step TD-errors for a minibatch of sampled experiences. The experienced SARST transition is
        added to the Experience buffer beforehand.

        Action-value estimates for all starting states are recomputed on the spot using the behaviour model.

        Args:
            action_values: (NotUsed)
            transition: SARST transition to add to the Experience buffer.

        Returns:
            One-step TD-errors for each sampled transition as a unidimensional tensor
        """
        self._store_transition(action_values, transition)
        batch, metadata = self.experience_buffer.sample(self.batch_size)
        # Compute TD-errors and apply weighted importance-sampling to compensate for the non-uniform sampling
        # probabilities. See [1]_ sect. 3.4 for more details.
        errors = self._compute_td_error_batch(batch)
        errors *= (len(self.experience_buffer) * metadata['probabilities']) ** -self.beta
        # Re-prioritise sampled transitions. Note, priorities are exponentiated before passing to priority buffer.
        updated_priorities = errors.abs() ** self.alpha
        self._experience_buffer.set_priority(metadata['indices'], updated_priorities)
        del batch
        del metadata
        return errors


class _DoubleDQNBase:
    """Deep Q-Network adapted to use double Q-learning as a way to mitigate the overestimation bias, introduced by
     maximising over the action-values of the target model, through resampling actions from the behaviour model.

    References:
        [1] Van Hasselt, H., Guez, A., & Silver, D. (2016). "Deep reinforcement learning with double q-learning". In
            Proceedings of the AAAI conference on artificial intelligence, 30.
    """
    behaviour_model: pt.nn.Module
    target_model: pt.nn.Module
    discount_factor: float

    def __init__(self):
        for requirement in ["behaviour_model", "target_model", "discount_factor"]:
            assert hasattr(self, requirement), \
                f'Double Q-learning requires a DQN implementation providing "{requirement}".'
        self._training_sampler = GreedyActionSampler()

    def _compute_td_error_batch(self, batch: t.Dict[str, pt.Tensor]) -> pt.Tensor:
        # Compute One-step TD-errors for all samples
        values = pt.take_along_dim(self.behaviour_model(batch['states']), batch['actions'].type(pt.long), dim=-1)[:, 0]
        successor_values = pt.take_along_dim(
            self.target_model(batch['next_states']),
            # Double Q-learning uses the behaviour model to select the best action instead of directly maximising over
            # the target network's action-value estimates, which are known to result in overestimation.
            self._training_sampler.sample(self.behaviour_model(batch['next_states']))[:, None].type(pt.long),
            dim=-1
        )
        return onestep_td_error(self.discount_factor, values, batch['rewards'], successor_values, batch['terminations'])


class DoubleDQN(DQNExperienceReplay, _DoubleDQNBase):
    """Deep Q-Network adapted to use double Q-learning as a way to mitigate the overestimation bias, introduced by
     maximising over the action-values of the target model, through resampling actions from the behaviour model.

    References:
        [1] Van Hasselt, H., Guez, A., & Silver, D. (2016). "Deep reinforcement learning with double q-learning". In
            Proceedings of the AAAI conference on artificial intelligence, 30.
    """
    def __init__(self,
                 action_value_model: ActionValueFunction,
                 *,
                 learning_rate: float,
                 discount_factor: float,
                 epsilon_greedy: float,
                 target_refresh_steps: int,
                 replay_memory_capacity: int,
                 batch_size: int):
        DQNExperienceReplay.__init__(
            self, action_value_model, learning_rate=learning_rate, discount_factor=discount_factor,
            epsilon_greedy=epsilon_greedy, target_refresh_steps=target_refresh_steps,
            replay_memory_capacity=replay_memory_capacity, batch_size=batch_size)
        _DoubleDQNBase.__init__(self)


class PrioritisedDoubleDQN(DQNPrioritisedExperienceReplay, _DoubleDQNBase):
    """The original Deep Q-Network with Prioritised Experience Replay proposed by [1]_. Incorporates double Q-learning
    (see [2]_) as a way to mitigate the overestimation bias, introduced by maximising over the action-values of the
    target model, through resampling actions from the behaviour model.

    References:
        [1] Schaul, T., Quan, J., Antonoglou, I., & et al. (2015). "Prioritized experience replay". arXiv:1511.05952.
        [2] Van Hasselt, H., Guez, A., & Silver, D. (2016). "Deep reinforcement learning with double q-learning". In
            Proceedings of the AAAI conference on artificial intelligence, 30.
    """
    def __init__(self,
                 action_value_model: ActionValueFunction,
                 *,
                 learning_rate: float,
                 discount_factor: float,
                 epsilon_greedy: float,
                 target_refresh_steps: int,
                 replay_memory_capacity: int,
                 batch_size: int,
                 alpha: float,
                 beta: float):
        DQNPrioritisedExperienceReplay.__init__(
            self, action_value_model, learning_rate=learning_rate, discount_factor=discount_factor,
            epsilon_greedy=epsilon_greedy, target_refresh_steps=target_refresh_steps,
            replay_memory_capacity=replay_memory_capacity, batch_size=batch_size, alpha=alpha, beta=beta)
        _DoubleDQNBase.__init__(self)

