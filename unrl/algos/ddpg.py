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
import torch.nn.functional as F

import unrl.types as t
from unrl.basic import onestep_td_error, mse
from unrl.config import validate_config
from unrl.containers import ContextualTransition, ContextualTrajectory
from unrl.experience_buffer import ExperienceBuffer
from unrl.functions import ContinuousPolicy, StateValueFunction, ContinuousActionValueFunction, GaussianPolicy
from unrl.optim import multi_optimiser_stepper, polyak_averaging_inplace
from unrl.utils import persisted_generator_value

__all__ = [
    "DDPG",
    "TwinDelayedDDPG",
    "TD3",
    "SAC",
    "QSAC"
]


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
    def optimise_online(
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
    def optimise_online(
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


TD3 = TwinDelayedDDPG


class SAC:
    """Soft Actor-Critic is an off-policy algorithm for continuous State and Action spaces. SAC optimises the "Maximum
    Entropy Objective", this objective expands the traditional Reinforcement Learning objective of maximising the sum of
    future discounted rewards with an Entropy term on the learnt Policy.

    Note this implementation follows the original SAC algorithm from [1]_ where explicitly training a soft State-value
    function alongside the Policy and twin Action-value functions is recommended. Their argument is doing so stabilises
    training (cf. Sect.4.2 [1]_). The authors shortly after walk back this claim in [2]_ and reformulate SAC without a
    State-value function. Choose class "QSAC" to avoid State-value functions.

    References:
        [1] Haarnoja, T., Zhou, A., Abbeel, P., & et al. (2018). "Soft actor-critic: Off-policy maximum entropy deep
            reinforcement learning with a stochastic actor". In Proceedings of the 35th International Conference on
            Machine Learning.
        [2] Haarnoja, T., Zhou, A., Hartikainen, K., & et al. (2018). "Soft actor-critic algorithms and applications".
            arXiv:1812.05905.
        [3] Lillicrap, T. P., Hunt, J. J., Pritzel, A., & et al. (2015). "Continuous control with deep reinforcement
            learning". arXiv:1509.02971.
        [4] Fujimoto, S., Hoof, H., & Meger, D. (2018). "Addressing function approximation error in actor-critic
            methods". In Proceedings of the 35th International Conference on Machine Learning.
    """
    def __init__(self,
                 policy: GaussianPolicy,
                 state_value_model: StateValueFunction,
                 action_value_model: ContinuousActionValueFunction,
                 action_value_twin_model: ContinuousActionValueFunction,
                 discount_factor: float,
                 learning_rate_policy: float,
                 learning_rate_actions: float,
                 learning_rate_states: float,
                 polyak_factor: float,
                 replay_memory_capacity: int,
                 batch_size: int,
                 target_refresh_steps: int,
                 epsilon: float = 1e-8):
        validate_config(discount_factor, 'discount_factor', 'unit')
        validate_config(learning_rate_policy, 'learning_rate_policy', 'unitpositive')
        validate_config(learning_rate_actions, 'learning_rate_actions', 'unitpositive')
        validate_config(learning_rate_states, 'learning_rate_states', 'unitpositive')
        validate_config(polyak_factor, 'polyak_factor', 'unit')
        validate_config(replay_memory_capacity, 'replay_memory_capacity', 'positive')
        validate_config(batch_size, 'batch_size', 'positive')
        validate_config(target_refresh_steps, 'target_refresh_steps', 'positive')
        validate_config(epsilon, 'epsilon', 'positive')
        self.policy = policy
        self.state_value_model = state_value_model
        self.target_state_value_model = deepcopy(state_value_model)
        self.action_value_model = action_value_model
        self.action_value_twin_model = action_value_twin_model

        self.discount_factor = discount_factor
        self.learning_rate_policy = learning_rate_policy
        self.learning_rate_actions = learning_rate_actions
        self.learning_rate_states = learning_rate_states
        self.replay_memory_capacity = replay_memory_capacity
        self.batch_size = batch_size
        self.target_refresh_steps = target_refresh_steps
        self.eps = epsilon

        self._experience_buffer = ExperienceBuffer(maxlen=replay_memory_capacity)
        self._polyak_weights = [polyak_factor, 1 - polyak_factor]

        self._optim_policy = pt.optim.SGD(self.policy.parameters(), lr=self.learning_rate_policy)
        self._optim_values = pt.optim.SGD(self.state_value_model.parameters(), lr=self.learning_rate_states)
        self._optim_actions = pt.optim.SGD(self.action_value_model.parameters(), lr=self.learning_rate_actions)
        self._optim_actions_twin = pt.optim.SGD(self.action_value_twin_model.parameters(),
                                                lr=self.learning_rate_actions)
        self._stepper = multi_optimiser_stepper(
            self._optim_policy, self._optim_values, self._optim_actions, self._optim_actions_twin)
        self.__steps = 0

    def sample_action(self, state: pt.Tensor, noisy: bool = False) -> t.Tuple[pt.Tensor, pt.Tensor]:
        dist = self._build_policy_distribution(state)
        if noisy:
            action = dist.sample(state.shape[0])
            logprob = dist.log_prob(action)
        else:
            unbounded_action = dist.mean + dist.variance * pt.randn_like(state.shape[0])
            action = F.tanh(unbounded_action)
            logprob = dist.log_prob(unbounded_action) - pt.log(1 - action ** 2 + self.eps)
        return action, logprob

    def _build_policy_distribution(self, state: pt.Tensor) -> pt.distributions.Distribution:
        """Return a distribution parameterised by a variational Policy"""
        mu, sigma = self.policy(state)
        return pt.distributions.Normal(mu, sigma)

    @persisted_generator_value
    def optimise_online(
        self,
        starting_state: pt.Tensor
    ) -> t.Generator[t.IntLike, t.Tuple[t.IntLike, pt.Tensor, bool, bool], t.Tuple[ContextualTrajectory, float]]:
        """Apply policy for one episode before optimising value functions and Policy by replaying a minibatch of
        experience.

        Args:
            starting_state:
                State from which to begin the episode.

        Yields:
            Action chosen by Policy for the current state.

        Receives:
            Tuple (reward, successor, terminal, stop) containing the reward obtained by applying the action yielded, the
            resulting successor state, and whether this state is terminal. And, finally, an indication of whether to
            stop irrespective of the ending up in a terminal state.

        Returns:
            A tuple (trajectory, average_loss) containing the trajectory of the full episode and the average total error
            for the sampled minibatch.
        """
        episode = []
        state = starting_state
        stop = False
        # Run episode
        while not stop:
            action, _ = self.sample_action(state)
            reward, next_state, terminal, stop = yield action
            episode.append(ContextualTransition(state, action, reward, next_state, terminal))
            state = next_state
        # Optimise models over one minibatch
        batch_loss = self.optimise_minibatch()
        return episode, batch_loss

    def optimise_minibatch(self, batch_size: t.Optional[int] = None) -> float:
        batch_size = batch_size or self.batch_size
        batch, _ = self._experience_buffer.sample(batch_size)

        actions, logprobs = self.sample_action(batch['states'])
        action_values = self.action_value_model(batch['states'], actions)
        action_values_twin = self.action_value_twin_model(batch['states'], actions)
        success_state_values = self.target_state_value_model(batch['next_states'])

        # Compute Action-value losses: eq.9 `∇θ JQ(θ) = ∇θ Qθ (at, st) (Qθ (st, at) − r(st, at) − γV ¯ψ (st+1)`.
        loss_actions = mse(onestep_td_error(
            self.discount_factor, action_values, batch['rewards'], success_state_values, batch['terminations']))
        loss_actions_twin = mse(onestep_td_error(
            self.discount_factor, action_values_twin, batch['rewards'], success_state_values, batch['terminations']))

        # to optimise the State-value function and the Policy we select the lowest action-value estimates among the twin
        # Action-value functions (see Sect.4.2 [1]_).

        # Compute State-value loss: eq.6 `∇ψ JV (ψ) = ∇ψ Vψ (st) (Vψ (st) − Qθ (st, at) + log πφ(at|st))`.
        state_values = self.state_value_model(batch['states'])
        loss_states = mse(state_values - pt.minimum(action_values, action_values_twin) + logprobs)

        # Compute Policy loss using the reparameterisation trick `â~fφ(εt; st)`: eq.13 `∇φJπ (φ) = ∇φ log πφ(at|st) +
        # (∇ât log πφ(ât|st) − ∇ât Q(st, ât))∇φfφ(εt; st)`.
        noisy_actions, noisy_logprobs = self.sample_action(batch['states'], noisy=True)
        noisy_action_values = self.action_value_model(batch['states'], actions)
        noisy_action_values_twin = self.action_value_twin_model(batch['states'], actions)
        loss_policy = (noisy_logprobs - pt.minimum(noisy_action_values, noisy_action_values_twin)).mean()

        loss = loss_policy + loss_states + loss_actions + loss_actions_twin
        self._stepper(loss)

        self.__steps = (self.__steps + 1) % self.target_refresh_steps
        if self.__steps == 0:
            # Note: [1]_ authors use a different coefficient (`tau`) to control Polyak averaging that in addition is set
            # specifically for the behaviour model, differing from [2]_ and [3]_. We retain the notation of DDPG.
            polyak_averaging_inplace([self.target_state_value_model, self.state_value_model], self._polyak_weights)

        return loss.item()


class QSAC:
    """Off-policy Soft Actor-Critic for continuous State and Action spaces, without State-value functions.

    This is the implementation of the updated SAC algorithm from [1]_ which no longer relies on a State-value function.

    References:
        [1] Haarnoja, T., Zhou, A., Hartikainen, K., & et al. (2018). "Soft actor-critic algorithms and applications".
            arXiv:1812.05905.
    """
    def __init__(self,
                 policy: GaussianPolicy,
                 action_value_model: ContinuousActionValueFunction,
                 action_value_twin_model: ContinuousActionValueFunction,
                 discount_factor: float,
                 learning_rate_policy: float,
                 learning_rate_actions: float,
                 entropy_coefficient: float,
                 polyak_factor: float,
                 replay_memory_capacity: int,
                 batch_size: int,
                 target_refresh_steps: int,
                 epsilon: float = 1e-8):
        validate_config(discount_factor, 'discount_factor', 'unit')
        validate_config(learning_rate_policy, 'learning_rate_policy', 'unitpositive')
        validate_config(learning_rate_actions, 'learning_rate_actions', 'unitpositive')
        validate_config(entropy_coefficient, 'entropy_coefficient', 'unit')
        validate_config(polyak_factor, 'polyak_factor', 'unit')
        validate_config(replay_memory_capacity, 'replay_memory_capacity', 'positive')
        validate_config(batch_size, 'batch_size', 'positive')
        validate_config(target_refresh_steps, 'target_refresh_steps', 'positive')
        validate_config(epsilon, 'epsilon', 'positive')
        self.policy = policy
        self.action_value_model = action_value_model
        self.action_value_twin_model = action_value_twin_model
        self.target_action_value_model = deepcopy(action_value_model)
        self.target_action_value_twin_model = deepcopy(action_value_twin_model)

        self.discount_factor = discount_factor
        self.learning_rate_policy = learning_rate_policy
        self.learning_rate_actions = learning_rate_actions
        self.entropy_coefficient = entropy_coefficient
        self.replay_memory_capacity = replay_memory_capacity
        self.batch_size = batch_size
        self.target_refresh_steps = target_refresh_steps
        self.eps = epsilon

        self._experience_buffer = ExperienceBuffer(maxlen=replay_memory_capacity)
        self._polyak_weights = [polyak_factor, 1 - polyak_factor]

        self._optim_policy = pt.optim.SGD(self.policy.parameters(), lr=self.learning_rate_policy)
        self._optim_actions = pt.optim.SGD(self.action_value_model.parameters(), lr=self.learning_rate_actions)
        self._optim_actions_twin = pt.optim.SGD(self.action_value_twin_model.parameters(),
                                                lr=self.learning_rate_actions)
        self._stepper = multi_optimiser_stepper(self._optim_policy, self._optim_actions, self._optim_actions_twin)
        self.__steps = 0

    _build_policy_distribution = SAC._build_policy_distribution
    sample_action = SAC.sample_action
    optimise_online = SAC.optimise_online

    def _compute_stochastic_estimates(self, states: pt.Tensor, noisy: bool = False, use_targets: bool = False) -> pt.Tensor:
        """Compute Action-value estimates with an Entropy bonus from resampled actions for a batch of states.

        The state-wise lowest Action-value estimate among the twin functions is used.

        Args:
            states: states from which to stochastically sample actions and obtain current Action-value estimates.
            use_targets: whether to use the target or behaviour twin Action-value functions.

        Returns:
            Tensor of Action-value estimates for the provided batch.
        """
        actions, logprobs = self.sample_action(states, noisy=noisy)
        if use_targets:
            action_values = pt.minimum(
                self.target_action_value_model(states, actions),
                self.target_action_value_twin_model(states, actions)
            )
        else:
            action_values = pt.minimum(
                self.action_value_model(states, actions),
                self.action_value_twin_model(states, actions)
            )
        return action_values - self.entropy_coefficient * logprobs

    def optimise_minibatch(self, batch_size: t.Optional[int] = None) -> float:
        batch_size = batch_size or self.batch_size
        batch, _ = self._experience_buffer.sample(batch_size)

        # For resampled actions, SAC uses the lowest action-value estimates among the twin Action-value functions to
        # optimise Action-value functions and Policy.
        action_values = self.action_value_model(batch['states'], batch['actions'])
        action_values_twin = self.action_value_twin_model(batch['states'], batch['actions'])
        target_estimates = self._compute_stochastic_estimates(batch['next_states'], use_targets=True)

        # Actions are sampled using the reparameterisation trick `â~fφ(εt; st)` to provide differentiable probabilities.
        loss_actions = mse(onestep_td_error(
            self.discount_factor, action_values, batch['rewards'], target_estimates, batch['terminations']))
        loss_actions_twin = mse(onestep_td_error(
            self.discount_factor, action_values_twin, batch['rewards'], target_estimates, batch['terminations']))
        loss_policy = self._compute_stochastic_estimates(batch['states'], noisy=True).mean()
        loss = loss_policy + loss_actions + loss_actions_twin
        self._stepper(loss)

        self.__steps = (self.__steps + 1) % self.target_refresh_steps
        if self.__steps == 0:
            polyak_averaging_inplace([self.target_action_value_model, self.action_value_model], self._polyak_weights)
            polyak_averaging_inplace([self.target_action_value_twin_model, self.action_value_twin_model],
                                     self._polyak_weights)

        return loss.item()
