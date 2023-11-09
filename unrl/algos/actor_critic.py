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
import torch.nn.functional as F

import unrl.types as t
from unrl.action_sampling import ActionSampler, make_sampler, ActionSamplingMode
from unrl.basic import onestep_td_error, entropy, mse
from unrl.config import validate_config
from unrl.containers import Transition, Trajectory, FrozenTrajectory, ContextualTrajectory, ContextualTransition
from unrl.functions import Policy, StateValueFunction
from unrl.optim import EligibilityTraceOptimizer, multi_optimiser_stepper
from unrl.utils import persisted_generator_value

__all__ = [
    "ActorCritic",
    "EligibilityTraceActorCritic",
    "AdvantageActorCritic",
    "A2C",
    "ICM",
]


class ActorCritic:
    """One-Step Online Actor-Critic for episodic settings"""
    policy: Policy
    state_value_model: StateValueFunction
    sampler: ActionSampler  # Strategy for sampling actions from the policy

    def __init__(self,
                 policy: Policy,
                 state_value_model: StateValueFunction,
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
    def optimise_online(
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
    state_value_model: StateValueFunction
    sampler: ActionSampler

    _optim_policy: EligibilityTraceOptimizer
    _optim_values: EligibilityTraceOptimizer

    def __init__(self,
                 policy: Policy,
                 state_value_model: StateValueFunction,
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
    def optimise_online(
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


class AdvantageActorCritic:
    """Advantage Actor-Critic (A2C) for episodic settings. This is the synchronous variant of the Asynchronous Advantage
    Actor-Critic (A3C) model proposed in [1]_. Advantage Actor-Critic algorithms update policies using estimates of the
    Advantage from state-values learnt by the critic.

    References:
        [1] Mnih, V., Badia, A. P., Mirza, M., & et al. (2016). "Asynchronous methods for deep reinforcement learning".
            In Proceedings of The 33rd International Conference on Machine Learning.
    """
    policy: Policy
    state_value_model: StateValueFunction
    sampler: ActionSampler  # Strategy for sampling actions from the policy

    def __init__(self,
                 policy: Policy,
                 state_value_model: StateValueFunction,
                 learning_rate_policy: float,
                 learning_rate_values: float,
                 discount_factor: float,
                 entropy_coefficient: t.Optional[float] = None,
                 action_sampler: t.Optional[ActionSampler] = None):
        """Args:
            policy: Policy model to improve
            state_value_model: State-value model to improve
            learning_rate_policy: learning rate for Policy model
            learning_rate_values: learning rate for state-value model
            discount_factor: discounting rate for future rewards
            entropy_coefficient: (Optional) Coefficient for adding entropy regularisation. Disabled if not specified.
            action_sampler: (Optional) strategy for sampling actions from the policy. Defaults to Greedy sampling if not
                            provided.
        """
        validate_config(learning_rate_policy, "learning_rate_policy", "unitpositive")
        validate_config(learning_rate_values, "learning_rate_values", "unitpositive")
        validate_config(discount_factor, "discount_factor", "unitpositive")
        if entropy_coefficient is not None:
            validate_config(entropy_coefficient, "entropy_coefficient", "unit")
        self.policy = policy
        self.state_value_model = state_value_model
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate_policy
        self.learning_rate_values = learning_rate_values
        self.entropy_coefficient = entropy_coefficient
        self._sampler = action_sampler or make_sampler(ActionSamplingMode.EPSILON_GREEDY)
        self._optim_policy = pt.optim.SGD(self.policy.parameters(), lr=self.learning_rate)
        self._optim_values = pt.optim.SGD(self.state_value_model.parameters(), lr=self.learning_rate_values)
        self._stepper = multi_optimiser_stepper(self._optim_policy, self._optim_values)

    def sample(self, state: pt.Tensor) -> int:
        logprobs = self.policy(state)
        action = self._sampler.sample(logprobs)
        return action.item()

    def optimise(self, episode: ContextualTrajectory) -> float:
        total_loss = 0
        entropy_reg = 0
        R = pt.Tensor([0]) if episode[-1].terminates else self.state_value_model(episode[-1].next_state)
        for transition in episode[::-1]:
            (state, action, reward, next_state, _) = transition
            R = reward + self.discount_factor * R
            logprobs = self.policy(state)

            if self.entropy_coefficient:
                entropy_reg += entropy(logits=logprobs)

            estimate = self.state_value_model(state)
            advantage = (R - estimate)
            loss_policy = logprobs[action] * advantage
            loss_values = (advantage ** 2).mean()
            total_loss += loss_policy + loss_values

        if self.entropy_coefficient:
            total_loss += self.entropy_coefficient * entropy_reg / len(episode)
        self._stepper(total_loss)
        return total_loss.item()

    def optimise_batch(self, episodes: t.Sequence[FrozenTrajectory]):
        for episode in episodes:
            self.optimise(episode)

    @persisted_generator_value
    def optimise_online(
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
        logits = []
        entropy_reg = None
        state = starting_state
        stop = False

        # Run episode
        while not stop:
            logprobs = self.policy(state)
            action = self._sampler.sample(logprobs)
            logits.append(logprobs[action])
            # Share action
            reward, next_state, terminal, stop = yield action
            # Save transition
            episode.append(ContextualTransition(state, action.type(pt.int), reward, next_state, terminal))
            state = next_state

            if self.entropy_coefficient:
                H = entropy(logits=logprobs)
                if entropy_reg is None:
                    entropy_reg = H
                else:
                    entropy_reg += H

        # Accumulate gradients
        R = pt.Tensor([0]) if episode[-1].terminates else self.state_value_model(episode[-1].next_state)
        total_loss = 0
        for transition, logit in zip(episode[::-1], logits):
            R = transition.reward + self.discount_factor * R
            estimate = self.state_value_model(transition.state)
            advantage = R - estimate
            # Compute losses
            loss_policy = -logit * advantage
            loss_values = (advantage ** 2).mean()
            total_loss += loss_values + loss_policy

        if self.entropy_coefficient:
            total_loss += self.entropy_coefficient * entropy_reg / len(episode)

        # Apply updates
        self._stepper(total_loss)
        del logits
        return episode, total_loss.item()


A2C = AdvantageActorCritic


class ICM:
    """Model-based Advantage Actor-Critic model with an Intrinsic Curiosity Module (ICM) that learns forward and inverse
    models of the effects of actions in the world, for discrete Action space settings. From mismatches between
    self-supervised predictions from the forward and inverse models and observations, ICM provides an intrinsic reward
    to complement the environment's that promotes exploration.

    Note: ICM supports continuous Action space, this implementation is only adapted for discrete Action spaces.

    References:
        [1] Deepak Pathak, Pulkit Agrawal, Alexei A. Efros & et al. (2017). "Curiosity-driven Exploration by
            Self-supervised Prediction". Proceedings of the 34th International Conference on Machine Learning.
    """

    def __init__(self,
                 policy: Policy,
                 state_value_model: StateValueFunction,
                 state_encoder: pt.nn.Module,
                 forward_model: pt.nn.Module,
                 inverse_model: pt.nn.Module,
                 learning_rate_policy: float,
                 learning_rate_values: float,
                 learning_rate_forward: float,
                 learning_rate_inverse: float,
                 discount_factor: float,
                 forward_loss_scaling: float,
                 policy_loss_importance: float,
                 beta: float,
                 action_sampler: t.Optional[ActionSampler] = None
                 ):
        self.policy = policy
        self.state_value_model = state_value_model
        self.state_encoder = state_encoder
        self.forward_model = forward_model
        self.inverse_model = inverse_model
        self.discount_factor = discount_factor
        self.forward_loss_scaling = forward_loss_scaling
        self.beta = beta
        self.policy_loss_importance = policy_loss_importance
        self._sampler = action_sampler or make_sampler(ActionSamplingMode.EPSILON_GREEDY)
        self._optim_policy = pt.optim.SGD(self.policy.parameters(), lr=learning_rate_policy)
        self._optim_values = pt.optim.SGD(self.state_value_model.parameters(), lr=learning_rate_values)
        self._optim_forward = pt.optim.SGD(self.forward_model.parameters(), lr=learning_rate_forward)
        self._optim_inverse = pt.optim.SGD(self.inverse_model.parameters(), lr=learning_rate_inverse)
        self._stepper = multi_optimiser_stepper(self._optim_policy, self._optim_values, self._optim_inverse)
        self._stepper_forward = multi_optimiser_stepper(self._optim_forward)

    sample = AdvantageActorCritic.sample

    def _compute_losses_and_running_reward(self,
                                           transition: ContextualTransition,
                                           logprobs: pt.Tensor,
                                           future_reward: pt.Tensor
                                           ) -> t.Tuple[pt.Tensor, pt.Tensor, pt.Tensor]:
        """Compute all losses and future reward to backpropagate

        Args:
            transition: SARST transition.
            logprobs: action log-probabilities conditioned on the transition's starting state.
            future_reward: total future reward from the subsequent transition.

        Returns:
            A tuple (total loss, forward model loss, reward from starting state). Where the total loss includes the
            forward loss.
        """
        # Encoded states
        encoded_state = self.state_encoder(transition.state)
        encoded_next_state = self.state_encoder(transition.next_state)
        # Inverse model's prediction of applied action
        inverse_action = self.inverse_model(encoded_state, encoded_next_state)
        inverse_error = F.cross_entropy(
            F.log_softmax(logprobs, dim=-1),
            F.softmax(F.one_hot(inverse_action, logprobs.shape[0]), dim=-1)
        )
        # Forward model's prediction of state transition
        predicted_next_state = self.forward_model(encoded_state, transition.action)
        forward_error_norm = mse(predicted_next_state - transition.state)
        # Set discounted future rewards and add intrinsic reward derived from the forward model's expectation
        intrinsic_reward = self.forward_loss_scaling * forward_error_norm
        reward = intrinsic_reward + transition.reward + self.discount_factor * future_reward
        # Calculate the advantage
        estimate = self.state_value_model(transition.state)
        advantage = reward - estimate
        # Compute losses
        loss_inverse = (1 - self.beta) * inverse_error
        loss_forward = forward_error_norm  # scaled by `beta` when passing to the joint loss
        loss_policy = self.policy_loss_importance * -logprobs[transition.action] * advantage
        loss_values = (advantage ** 2).mean()
        total_loss = loss_values + loss_policy + self.beta * loss_forward + loss_inverse
        return total_loss, loss_forward, reward

    def optimise(self, episode: ContextualTrajectory) -> float:
        total_loss = total_forward_loss = 0
        R = pt.Tensor([0]) if episode[-1].terminates else self.state_value_model(episode[-1].next_state)
        for transition in episode[::-1]:
            logprobs = self.policy(transition.state)
            transition_loss, forward_loss, R = self._compute_losses_and_running_reward(transition, logprobs, R)
            total_loss += transition_loss
            total_forward_loss += forward_loss

        # Apply updates. Note that policy loss gradient must not be backpropagated to the forward model to prevent
        # degenerate solutions where the agent rewards itself (see Sect.2.2 [1]_).
        self._stepper_forward(total_forward_loss, retain_graph=True)
        self._stepper(total_loss)
        return total_loss.item()

    optimise_batch = AdvantageActorCritic.optimise_batch

    @persisted_generator_value
    def optimise_online(
        self,
        starting_state: pt.Tensor,
    ) -> t.Generator[t.IntLike, t.Tuple[t.IntLike, pt.Tensor, bool, bool], t.Tuple[Trajectory, float]]:
        """Optimise policy, state-value function, and world models online for one episode.

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
        logits = []
        state = starting_state
        stop = False

        # Run episode
        while not stop:
            logprobs = self.policy(state)
            action = self._sampler.sample(logprobs)
            logits.append(logprobs)
            # Share action
            reward, next_state, terminal, stop = yield action
            # Save transition
            episode.append(ContextualTransition(state, action.type(pt.int), reward, next_state, terminal))
            state = next_state

        # Accumulate gradients
        R = pt.Tensor([0]) if episode[-1].terminates else self.state_value_model(episode[-1].next_state)
        total_loss = total_forward_loss = 0
        for transition, logprobs in zip(episode[::-1], logits):
            transition_loss, forward_loss, R = self._compute_losses_and_running_reward(transition, logprobs, R)
            total_loss += transition_loss
            total_forward_loss += forward_loss

        # Apply updates. Note that policy loss gradient must not be backpropagated to the forward model to prevent
        # degenerate solutions where the agent rewards itself (see Sect.2.2 [1]_).
        self._stepper_forward(total_forward_loss, retain_graph=True)
        self._stepper(total_loss)
        del logits
        return episode, total_loss.item()
