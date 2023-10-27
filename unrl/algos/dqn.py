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
from copy import deepcopy

import torch as pt
from torch.optim import SGD

import unrl.types as t
from unrl.config import validate_config
from unrl.action_sampling import EpsilonGreedyActionSampler
from unrl.containers import ContextualTransition, ContextualTrajectory
from unrl.utils import persisted_generator_value


class DQN:
    """Basic Deep Q-Network (without an Experience Replay buffer), adapted from [1]_.

    References:
        [1] Mnih, V. & Kavukcuoglu, K. & Silver, D., et al. (2015). "Human-level control through deep reinforcement
            learning". Nature, 518, pp. 529-533.
    """
    def __init__(self,
                 action_value_model: pt.nn.Module,
                 *,
                 learning_rate: float,
                 discount_factor: float,
                 epsilon_greedy: float,
                 target_refresh_steps: int):
        super().__init__()
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
    ) -> t.Generator[t.IntLike, t.Tuple[t.IntLike, pt.Tensor, bool], t.Tuple[ContextualTrajectory, float]]:
        """Optimise policy and state-value function online for one episode until termination (as determined by caller,
        can include e.g. a forced stoppage after a certain number of episodes).

        Args:
            starting_state:
                State from which to begin the episode.

        Yields:
            Action chosen by policy for the current state.

        Receives:
            Tuple (reward, successor, terminal) containing the reward obtained by applying the action yielded, the
            resulting successor state, and whether this state is terminal.

        Returns:
            A tuple with a ContextualTrajectory of SARST transitions representing the full episode and the total loss
            of the experienced during the episode until termination.
        """
        episode = []
        total_loss = 0
        state = starting_state
        terminal = False
        step = 0
        while not terminal:
            step += 1
            action_values = self.behaviour_model(state)
            action = self._sampler.sample(action_values)

            reward, next_state, terminal = yield action

            transition = ContextualTransition(state, action.type(pt.int), reward, next_state, terminal)
            episode.append(transition)
            state = next_state

            error = self._compute_td_error(action_values, transition)
            total_loss += self._step(error)

            if step % self.target_refresh_steps == 0:
                self.target_model = deepcopy(self.behaviour_model)

        return episode, total_loss

    def _compute_td_error(self, action_values: pt.Tensor, transition: ContextualTransition) -> pt.Tensor:
        """Compute the One-step TD-error using the SARST transition and the current state's action-value estimates
        provided by the behaviour model."""
        if transition.terminates:
            target = pt.Tensor([0])
        else:
            target = self.discount_factor * self.target_model(transition.next_state).max()
        return target - action_values[transition.action]

    def _step(self, td: t.FloatLike) -> float:
        """Compute MSE loss from One-step TD-error to backpropagate gradients and update the behaviour model."""
        # Compute aggregated loss and backpropagate gradients
        loss = (td ** 2) / len(td)
        loss.backward()
        # Update parameters
        self._optim.step()
        self._optim.zero_grad()
        return loss.item()
