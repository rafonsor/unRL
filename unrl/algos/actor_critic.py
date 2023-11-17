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

import torch as pt
import torch.nn.functional as F

import unrl.types as t
from unrl.action_sampling import ActionSampler, make_sampler, ActionSamplingMode
from unrl.basic import onestep_td_error, entropy, mse, expected_value, rho_dists, rho_logits
from unrl.config import validate_config
from unrl.containers import SARSTransition, SARSTrajectory, FrozenTrajectory, ContextualTrajectory, \
    ContextualTransition, Transition, Trajectory
from unrl.experience_buffer import TrajectoryExperienceBuffer
from unrl.functions import Policy, StateValueFunction, ActionValueFunction, DuelingContinuousActionValueFunction, \
    GaussianPolicy
from unrl.optim import EligibilityTraceOptimizer, multi_optimiser_stepper, KFACOptimizer, polyak_averaging_inplace, \
    multi_optimiser_guard
from unrl.utils import persisted_generator_value

__all__ = [
    "ActorCritic",
    "EligibilityTraceActorCritic",
    "AdvantageActorCritic",
    "A2C",
    "ICM",
    "DiscreteACER",
    "ACER"
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
    ) -> t.Generator[t.IntLike, t.Tuple[t.IntLike, pt.Tensor, bool, bool], t.Tuple[SARSTrajectory, float]]:
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
            episode.append(SARSTransition(state, action.type(pt.int), reward, next_state))

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
    ) -> t.Generator[t.IntLike, t.Tuple[t.IntLike, pt.Tensor, bool, bool], t.Tuple[SARSTrajectory, float]]:
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
            episode.append(SARSTransition(state, action.type(pt.int), reward, next_state))
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
    ) -> t.Generator[t.IntLike, t.Tuple[t.IntLike, pt.Tensor, bool, bool], t.Tuple[SARSTrajectory, float]]:
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
    ) -> t.Generator[t.IntLike, t.Tuple[t.IntLike, pt.Tensor, bool, bool], t.Tuple[SARSTrajectory, float]]:
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


class ACKTR(AdvantageActorCritic):
    """Actor-Critic using Kronecker-factored Trust Region. This algorithm implements an Actor-Critic framework,
    optimised using the principles of TRPO: bounded updates following the natural gradient; that bring higher sample
    efficiency. Instead of directly calculating the natural gradient with the KL divergence between the behaviour and
    target Policies, ACKTR uses the Kronecker-factored Approximate Curvature method as an approximation. Parameter
    updates are further applied layer-by-layer to reduce the computational complexity of the factorisation.

    References:
        [1] Wu, Y., Mansimov, E., Grosse, R. B., & et al. (2017). "Scalable trust-region method for deep reinforcement
            learning using kronecker-factored approximation". Advances in neural information processing systems, 30.
    """
    def __init__(self,
                 policy: Policy,
                 state_value_model: StateValueFunction,
                 learning_rate_policy: float,
                 learning_rate_values: float,
                 discount_factor: float,
                 trust_region_radius: float = 1e-3,
                 entropy_coefficient: float = 1e-2,
                 ):
        validate_config(trust_region_radius, "trust_region_radius", "nonnegative")
        super().__init__(
            policy, state_value_model, learning_rate_policy, learning_rate_values, discount_factor, entropy_coefficient)
        self.trust_region_radius = trust_region_radius
        self._optim_policy = KFACOptimizer(self.policy, trust_region_radius=trust_region_radius,
                                           learning_rate=learning_rate_policy, max_learning_rate=learning_rate_policy)
        self._optim_values = KFACOptimizer(self.policy, trust_region_radius=trust_region_radius,
                                           learning_rate=learning_rate_values, max_learning_rate=learning_rate_values)
        self._stepper = multi_optimiser_stepper(self._optim_policy, self._optim_values)


class DiscreteACER:
    """Sample Efficient Actor-Critic with Experience Replay is a flexible Off-Policy algorithm proposed in [1]_ for
    discrete and continuous Action spaces. ACER builds on the ideas of TRPO ([2]_) and dueling Q-networks ([3]_) adapted
    for Actor-Critic settings. However, "DiscreteACER" specifically implements support for discrete Action spaces only
    by following Algorithm 2 of [1]_ to the letter. Importantly, there is no necessity to use a Stochastic Dueling
    Action-value Function (see Appendix A [1]_).

    References:
        [1] Wang, Z., Bapst, V., Heess, N., & et al. (2016). "Sample efficient actor-critic with experience replay".
            arXiv:1611.01224. (Accepted as a poster in ICLR 2017.)
        [2] Schulman, J., Levine, S., Abbeel, & et al. (2015). "Trust region policy optimization". In Proceedings of The
            32nd International Conference on Machine Learning.
        [3] Wang, Z., Schaul, T., Hessel, M., & et al. (2016). "Dueling network architectures for deep reinforcement
         learning". In Proceedings of The 33rd International Conference on Machine Learning.
    """
    def __init__(self,
                 policy: Policy,
                 action_value_model: ActionValueFunction,
                 learning_rate_policy: float,
                 learning_rate_values: float,
                 discount_factor: float,
                 polyak_factor: float,
                 importance_weight_truncation: float,
                 trust_region_radius: float,
                 replay_memory_capacity: int,
                 replay_rate: float,
                 ):
        validate_config(learning_rate_policy, "learning_rate_policy", "unitpositive")
        validate_config(learning_rate_values, "learning_rate_values", "unitpositive")
        validate_config(discount_factor, "discount_factor", "unitpositive")
        validate_config(polyak_factor, "polyak_factor", "unitpositive")
        validate_config(importance_weight_truncation, "importance_weight_truncation", "unitpositive")
        validate_config(trust_region_radius, "trust_region_radius", "positive")
        validate_config(replay_memory_capacity, "replay_memory_capacity", "positive")
        validate_config(replay_rate, "replay_rate", "nonnegative")

        self.policy = policy
        self.average_policy = deepcopy(policy)
        self.action_value_model = action_value_model
        self.learning_rate_policy = learning_rate_policy
        self.learning_rate_values = learning_rate_values
        self.discount_factor = discount_factor
        self.polyak_factor = polyak_factor
        self.importance_weight_truncation = importance_weight_truncation
        self.trust_region_radius = trust_region_radius
        self.replay_rate = replay_rate

        self.sampler = make_sampler(ActionSamplingMode.STOCHASTIC)
        self._replay_size_sampler = pt.Poisson(pt.Tensor((replay_rate,)))
        self._experience_buffer = TrajectoryExperienceBuffer(maxlen=replay_memory_capacity)
        self._optim_policy = pt.optim.SGD(self.policy.parameters(), lr=learning_rate_policy)
        self._optim_values = pt.optim.SGD(self.action_value_model.parameters(), lr=learning_rate_values)

    def optimise_batch(self, episodes: t.Sequence[Trajectory]) -> float:
        return sum(self.optimise(episode) for episode in episodes) / len(episodes)

    def optimise(self, episode: Trajectory) -> float:
        return self._acer(episode)

    @persisted_generator_value
    def optimise_online(
        self,
        starting_state: pt.Tensor,
    ) -> t.Generator[t.IntLike, t.Tuple[t.IntLike, pt.Tensor, bool, bool], t.Tuple[Trajectory, float]]:
        """Optimise Policy and Action-value function online for one on-policy episode and "n" off-policy episodes. The
        number of off-policy episodes is determined through Poisson sampling as configured by "_replay_size_sampler".

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
        state = starting_state
        stop = False
        while not stop:
            logprobs: pt.Tensor = self.policy(state)
            action = self.sampler.sample(logprobs)
            reward, next_state, terminal, stop = yield action
            # Save transition
            episode.append(Transition(state, action.type(pt.int), reward, next_state, terminal, logprobs.detach()))
            state = next_state
        onpolicy_loss = self._acer(episode, on_policy=True)

        # store episode in buffer
        self._experience_buffer.append(episode)

        # Sample "n" trajectories to train as off-policy episodes
        n = self._replay_size_sampler.sample().item()
        offpolicy_loss = self.optimise_minibatch(n)

        total_loss = (onpolicy_loss + n*offpolicy_loss) / (n + 1)
        return episode, total_loss

    def optimise_minibatch(self, batch_size: int) -> float:
        episodes, _ = self._experience_buffer.sample(batch_size)
        return self.optimise_batch(episodes)

    def _acer(self, trajectory: Trajectory, on_policy: bool = False) -> float:
        with multi_optimiser_guard(self._optim_policy, self._optim_values):
            episode_loss = self._acer_innerloop(trajectory, on_policy=on_policy)
        polyak_averaging_inplace([self.average_policy, self.policy], [self.polyak_factor, 1 - self.polyak_factor])
        return episode_loss

    def _acer_innerloop(self, trajectory: Trajectory, on_policy: bool) -> float:
        running_policy_loss = 0
        running_values_loss = 0
        # Initialise retraced Q-value
        last_transition = trajectory[-1]
        if last_transition.terminates:
            retraced_action_value = 0
        else:
            action_values = self.action_value_model(last_transition.state, combine=True)
            logprobs = self.policy(last_transition.state)
            retraced_action_value = expected_value(action_values, logits=logprobs)
            del action_values, logprobs
        del last_transition

        for transition in trajectory[::-1]:
            state, action, reward, next_state, terminal, behaviour_logprobs = transition
            logprobs = behaviour_logprobs if on_policy else self.policy(state)
            average_logprobs = self.average_policy(state)
            action_values = self.action_value_model(state)

            retraced_action_value = reward + self.discount_factor * retraced_action_value
            # Note, ACER for discrete Action spaces as defined in [1]_'s Algorithm 2 does not use state-value estimates
            # from the Dueling Action-value function. It computes instead, an expectation of action-value estimates over
            # the action distribution given by the latest policy.
            state_value = expected_value(action_values, logits=logprobs)

            # Unlike for continuous Action spaces, importance weights here are not scaled by the inverse dimensionality
            # of the Action space.
            importance_weight = rho_logits(logprobs, behaviour_logprobs, action)
            truncated_importance_weight = min(self.importance_weight_truncation, importance_weight)
            bounded_importance_weights = pt.clip(
                1 - (self.importance_weight_truncation * behaviour_logprobs) / logprobs, min=0)

            # Compute Trust Region quantities. Note "g" and "k" must still be derived wrt Policy to constrain updates.
            g = (
                truncated_importance_weight * logprobs[action] * (retraced_action_value - state_value)
                + (bounded_importance_weights * logprobs.exp() * logprobs * (action_values - state_value)).sum(dim=-1)
            )
            k = pt.kl_div(average_logprobs, logprobs, log_target=True)
            running_policy_loss += g.item() + k.item()

            # Compute Action-value function loss
            delta = (retraced_action_value - action_values[action])
            loss_values = delta ** 2
            running_values_loss += loss_values.item()

            # Accumulate gradients
            self._commit_policy_gradients(g, k)
            loss_values.backward()

            retraced_action_value = min(1, importance_weight) * delta + state_value
        return running_policy_loss + running_values_loss

    def _commit_policy_gradients(self, g, k):
        ggrads = pt.autograd.grad(g, self.policy.parameters())
        kgrads = pt.autograd.grad(k, self.policy.parameters())
        with pt.no_grad():
            for param, dg, dk in zip(self.policy.parameters(), ggrads, kgrads):
                # Bound all gradients to Trust Region
                grad = dg - max(0, (dk.T @ dg - self.trust_region_radius) / pt.linalg.norm(dk) ** 2)
                if param.grad is None:
                    param.grad = grad
                else:
                    param.grad += grad


class ACER(DiscreteACER):
    """Sample Efficient Actor-Critic with Experience Replay is a flexible Off-Policy algorithm, here implemented for
    continuous Action spaces parameterised as a Gaussian distribution. Builds on the ideas of TRPO ([2]_) and dueling
    Q-networks ([3]_) adapted for Actor-Critic settings.

    References:
        [1] Wang, Z., Bapst, V., Heess, N., & et al. (2016). "Sample efficient actor-critic with experience replay".
            arXiv:1611.01224. (Accepted as a poster in ICLR 2017.)
        [2] Schulman, J., Levine, S., Abbeel, & et al. (2015). "Trust region policy optimization". In Proceedings of The
            32nd International Conference on Machine Learning.
        [3] Wang, Z., Schaul, T., Hessel, M., & et al. (2016). "Dueling network architectures for deep reinforcement
            learning". In Proceedings of The 33rd International Conference on Machine Learning.
    """
    def __init__(self,
                 policy: GaussianPolicy,
                 action_value_model: DuelingContinuousActionValueFunction,
                 learning_rate_policy: float,
                 learning_rate_values: float,
                 discount_factor: float,
                 polyak_factor: float,
                 importance_weight_truncation: float,
                 trust_region_radius: float,
                 replay_memory_capacity: int,
                 replay_rate: float,
                 num_action_samples: int,
                 ):
        super().__init__(policy, action_value_model, learning_rate_policy, learning_rate_values, discount_factor,
                         polyak_factor, importance_weight_truncation, trust_region_radius, replay_memory_capacity,
                         replay_rate)
        validate_config(num_action_samples, "num_action_samples", "positive")
        self.num_action_samples = num_action_samples

    def _acer_innerloop(self, trajectory: Trajectory, on_policy: bool) -> float:
        running_policy_loss = 0
        running_values_loss = 0
        # Initialise retraced Q-value
        last_transition = trajectory[-1]
        if last_transition.terminates:
            retraced_action_value = 0
        else:
            state_value, _ = self.action_value_model(last_transition.state, combine=False)
            retraced_action_value = state_value.detach()
            del state_value
        del last_transition
        corrected_action_value = retraced_action_value

        for transition in trajectory[::-1]:
            retraced_action_value = transition.reward + self.discount_factor * retraced_action_value
            corrected_action_value = transition.reward + self.discount_factor * corrected_action_value

            # Retrieve state- and action-value estimates for original action
            dist = transition.dist if on_policy else self.policy(transition.state, dist=True)
            stochastic_actions = pt.split(dist.sample(pt.Size((self.num_action_samples,))), self.num_action_samples)
            state_value, advantage, expected = self.action_value_model(transition.state, transition.action,
                                                                       stochastic_actions, combine=False)
            stochastic_action_value = state_value + advantage - expected
            del advantage, expected
            # Compute action-value estimate for a newly sampled action
            new_action = dist.sample()
            stochastic_new_action_value = self.action_value_model(transition.state, new_action, stochastic_actions)
            del stochastic_actions

            # Compute all importance weights for original action and newly sampled action.
            rho = rho_dists(dist, transition.dist, transition.action)
            bounded_rho = min(1, rho)
            truncated_rho = min(self.importance_weight_truncation, rho)
            new_rho = rho_dists(dist, transition.dist, new_action)
            truncated_new_rho = max(0, 1 - (self.importance_weight_truncation / new_rho))

            # Compute Trust Region quantities. Note "g" and "k" must still be derived wrt Policy to constrain updates.
            g = (
                truncated_rho * dist.log_prob(transition.action) * (corrected_action_value - state_value)
                + truncated_new_rho * dist.log_prob(new_action) * (stochastic_new_action_value - state_value)
            )
            k = pt.distributions.kl_divergence(self.average_policy(transition.state, dist=True), dist)
            running_policy_loss += g.item() + k.item()

            # Compute Action-value function loss
            delta = (retraced_action_value - stochastic_action_value).detach()
            loss_values = delta * stochastic_action_value + bounded_rho * delta * state_value
            running_values_loss += loss_values.item()

            # Accumulate gradients
            self._commit_policy_gradients(g, k)
            loss_values.backward()

            # With continuous Action spaces, the retraced value scales importance weights by Action space dimension.
            retraced_action_value = bounded_rho ** (1 / transition.action.numel()) * delta + state_value
            corrected_action_value = corrected_action_value - stochastic_action_value + state_value
        return running_policy_loss + running_values_loss
