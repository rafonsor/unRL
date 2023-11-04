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
from unrl.action_sampling import ActionSamplingMode, make_sampler
from unrl.algos.actor_critic import ActorCritic, EligibilityTraceActorCritic, AdvantageActorCritic
from unrl.algos.dqn import DQN, DQNExperienceReplay, DoubleDQN, PrioritisedDoubleDQN, DQNPrioritisedExperienceReplay
from unrl.algos.policy_gradient import Reinforce, BaselineReinforce
from unrl.algos.ddpg import DDPG, TwinDelayedDDPG, SAC, QSAC
from unrl.functions import Policy, ContinuousPolicy, GaussianPolicy, ContinuousActionValueFunction, ActionValueFunction, \
    StateValueFunction


class ExamplePolicy(Policy):
    def __init__(self, num_state_dims: int, num_actions: int, hidden_dim: int):
        super().__init__()
        self.layer1 = pt.nn.Linear(num_state_dims, hidden_dim)
        self.layer2 = pt.nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = pt.nn.Linear(hidden_dim, num_actions)
        self.logsm = pt.nn.LogSoftmax(0)

    def forward(self, state: pt.Tensor) -> pt.Tensor:
        x = F.relu(self.layer1(state))
        x = F.relu(self.layer2(x))
        return self.logsm(self.layer3(x))


class ExampleContinuousPolicy(ContinuousPolicy):
    def __init__(self, num_state_dims: int, num_action_dims: int, hidden_dim: int):
        super().__init__()
        self.layer1 = pt.nn.Linear(num_state_dims, hidden_dim)
        self.layer2 = pt.nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = pt.nn.Linear(hidden_dim, num_action_dims)

    def forward(self, state: pt.Tensor) -> pt.Tensor:
        x = F.relu(self.layer1(state))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


class ExampleGaussianPolicy(GaussianPolicy):
    def __init__(self, num_state_dims: int, num_action_dims: int, hidden_dim: int):
        super().__init__()
        self.layer1 = pt.nn.Linear(num_state_dims, hidden_dim)
        self.layer2 = pt.nn.Linear(hidden_dim, hidden_dim)
        self.layer_mu = pt.nn.Linear(hidden_dim, num_action_dims)
        self.layer_sigma = pt.nn.Linear(hidden_dim, num_action_dims * num_action_dims)
        self.num_action_dims = num_action_dims

    def forward(self, state: pt.Tensor) -> t.Tuple[pt.Tensor, pt.Tensor]:
        x = F.relu(self.layer1(state))
        x = F.relu(self.layer2(x))
        mu = self.layer_mu(x)
        sigma = F.relu(self.layer_sigma(x)).reshape(self.num_action_dims, self.num_action_dims)
        return mu, sigma


class ExampleStateValueModel(StateValueFunction):
    def __init__(self, num_state_dims, hidden_dim: int):
        super().__init__()
        self.layer1 = pt.nn.Linear(num_state_dims, hidden_dim)
        self.layer2 = pt.nn.Linear(hidden_dim, hidden_dim // 3)
        self.layer3 = pt.nn.Linear(hidden_dim // 3, 1)

    def forward(self, state: pt.Tensor) -> pt.Tensor:
        x = F.relu(self.layer1(state))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


class ExampleActionValueEstimator(ActionValueFunction):
    def __init__(self, num_state_dims: int, num_actions: int, hidden_dim: int):
        super().__init__()
        self.layer1 = pt.nn.Linear(num_state_dims, hidden_dim)
        self.layer2 = pt.nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = pt.nn.Linear(hidden_dim, num_actions)

    def forward(self, state: pt.Tensor) -> pt.Tensor:
        x = F.relu(self.layer1(state))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


class ExampleContinuousActionValueEstimator(ContinuousActionValueFunction):
    def __init__(self, num_state_dims: int, num_action_dims: int, hidden_dim: int):
        super().__init__()
        self.layer_state = pt.nn.Linear(num_state_dims, hidden_dim)
        self.layer_action = pt.nn.Linear(num_action_dims, hidden_dim)
        self.layer2 = pt.nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = pt.nn.Linear(hidden_dim, 1)

    def forward(self, state: pt.Tensor, action: pt.Tensor) -> pt.Tensor:
        x1 = F.relu(self.layer_state(state))
        x2 = F.relu(self.layer_action(action))
        x = F.relu(self.layer2(x1 + x2))
        return self.layer3(x)


def prepare_game_model_reinforce(num_state_dims: int, num_actions: int, baseline: bool = False) -> Reinforce:
    discount_factor = 0.9
    learning_rate_policy = 1e-4
    learning_rate_values = 1e-4
    hidden_dim_policy = 150
    hidden_dim_values = 180
    policy = ExamplePolicy(num_state_dims, num_actions, hidden_dim_policy)
    if baseline:
        state_value_model = ExampleStateValueModel(num_state_dims, hidden_dim_values)
        reinforce = BaselineReinforce(policy, state_value_model,
                                      discount_factor=discount_factor,
                                      learning_rate_policy=learning_rate_policy,
                                      learning_rate_values=learning_rate_values)
    else:
        reinforce = Reinforce(policy, discount_factor=discount_factor, learning_rate=learning_rate_policy)
    return reinforce


def prepare_game_model_actorcritic(num_state_dims: int,
                                   num_actions: int,
                                   advantage: bool = False,
                                   eligibility_traces: bool = False
                                   ) -> ActorCritic:
    discount_factor = 0.9
    learning_rate_policy = 1e-8
    learning_rate_values = 1e-8
    trace_decay_policy = 0.10
    trace_decay_values = 0.10
    hidden_dim_policy = 10
    hidden_dim_values = 30
    weight_decay_policy = 0.1
    weight_decay_values = 0.1
    entropy_coefficient = 0.2
    policy = ExamplePolicy(num_state_dims, num_actions, hidden_dim_policy)
    state_value_model = ExampleStateValueModel(num_state_dims, hidden_dim_values)
    action_sampler = make_sampler(ActionSamplingMode.EPSILON_GREEDY)
    if advantage:
        actor_critic = AdvantageActorCritic(
            policy, state_value_model,
            discount_factor=discount_factor,
            learning_rate_policy=learning_rate_policy,
            learning_rate_values=learning_rate_values,
            entropy_coefficient=entropy_coefficient,
            action_sampler=action_sampler)
    elif eligibility_traces:
        actor_critic = EligibilityTraceActorCritic(
            policy, state_value_model,
            discount_factor=discount_factor,
            learning_rate_policy=learning_rate_policy,
            learning_rate_values=learning_rate_values,
            trace_decay_policy=trace_decay_policy,
            trace_decay_values=trace_decay_values,
            weight_decay_policy=weight_decay_policy,
            weight_decay_values=weight_decay_values,
            action_sampler=action_sampler)
    else:
        actor_critic = ActorCritic(
            policy, state_value_model, discount_factor=discount_factor, learning_rate_policy=learning_rate_policy,
            learning_rate_values=learning_rate_values, action_sampler=action_sampler)
    return actor_critic


def prepare_game_model_dqn(num_state_dims: int, num_actions: int, buffer_size: t.Optional[int] = None, priority: bool = False, double: bool = False) -> DQN:
    discount_factor = 0.99
    learning_rate = 1e-2
    epsilon_greedy = 0.1
    refresh_steps = 1000
    hidden_dim = 60
    replay_minibatch = 64
    per_alpha = 0.7
    per_beta = 0.5
    action_value_model = ExampleActionValueEstimator(num_state_dims, num_actions, hidden_dim)
    model_kwargs = {
        "learning_rate": learning_rate,
        "discount_factor": discount_factor,
        "epsilon_greedy": epsilon_greedy,
        "target_refresh_steps": refresh_steps,
        "replay_memory_capacity": buffer_size,
        "batch_size": replay_minibatch
    }
    if buffer_size:
        model_kwargs.update({
            "replay_memory_capacity": buffer_size,
            "batch_size": replay_minibatch
        })
        if priority:
            model_kwargs.update({
                "alpha": per_alpha,
                "beta": per_beta
            })
            if double:
                dqn = PrioritisedDoubleDQN
            else:
                dqn = DQNPrioritisedExperienceReplay
        else:
            if double:
                dqn = DoubleDQN
            else:
                dqn = DQNExperienceReplay
    else:
        dqn = DQN
    return dqn(action_value_model, **model_kwargs)


def prepare_game_model_ddpg(num_state_dims: int, num_action_dims: int, twin: bool = False) -> DDPG:
    discount_factor = 0.9
    learning_rate_policy = 1e-4
    learning_rate_values = 1e-4
    hidden_dim_policy = 150
    hidden_dim_values = 180
    replay_memory_capacity = 10000
    replay_minibatch = 64
    noise_scale = 0.1
    noise_exploration = 0.05
    noise_epsilon = 0.1
    polyak_factor = 0.9
    refresh_steps = 1000
    policy_update_delay = 2
    policy = ExampleContinuousPolicy(num_state_dims, num_action_dims, hidden_dim_policy)
    action_value_model = ExampleContinuousActionValueEstimator(num_state_dims, num_action_dims, hidden_dim_values)
    model_kwargs = {
        "discount_factor": discount_factor,
        "learning_rate_policy": learning_rate_policy,
        "learning_rate_values": learning_rate_values,
        "noise_scale": noise_scale,
        "noise_exploration": noise_exploration,
        "polyak_factor": polyak_factor,
        "replay_memory_capacity": replay_memory_capacity,
        "batch_size": replay_minibatch,
        "target_refresh_steps": refresh_steps
    }
    if twin:
        action_value_twin_model = ExampleContinuousActionValueEstimator(num_state_dims, num_action_dims,
                                                                        hidden_dim_values)
        model_kwargs.update({
            "noise_epsilon": noise_epsilon,
            "policy_update_delay": policy_update_delay
        })
        ddpg = TwinDelayedDDPG(policy, action_value_model, action_value_twin_model, **model_kwargs)
    else:
        ddpg = DDPG(policy, action_value_model, **model_kwargs)
    return ddpg


def prepare_game_model_sac(num_state_dims: int, num_action_dims: int, use_state_value_function: bool = True) -> SAC:
    discount_factor = 0.99
    learning_rate_policy = 3e-4
    learning_rate_values = 3e-4
    learning_rate_actions = 3e-4
    hidden_dim_policy = 256
    hidden_dim_values = 256
    hidden_dim_actions = 256
    replay_memory_capacity = 10000
    replay_minibatch = 256
    polyak_factor = 0.995
    entropy_coefficient = -num_action_dims
    refresh_steps = 1  # The target State-value function is continuously refreshed
    model_kwargs = {
        "discount_factor": discount_factor,
        "learning_rate_policy": learning_rate_policy,
        "learning_rate_values": learning_rate_values,
        "learning_rate_actions": learning_rate_actions,
        "polyak_factor": polyak_factor,
        "replay_memory_capacity": replay_memory_capacity,
        "batch_size": replay_minibatch,
        "target_refresh_steps": refresh_steps
    }
    policy = ExampleGaussianPolicy(num_state_dims, num_action_dims, hidden_dim_policy)
    action_value_model = ExampleContinuousActionValueEstimator(num_state_dims, num_action_dims, hidden_dim_actions)
    action_value_twin_model = ExampleContinuousActionValueEstimator(num_state_dims, num_action_dims, hidden_dim_actions)
    if use_state_value_function:
        model_kwargs.update({
            "learning_rate_values": learning_rate_values,
        })
        state_value_model = ExampleStateValueModel(num_state_dims, hidden_dim_values)
        sac = SAC(policy, state_value_model, action_value_model, action_value_twin_model, **model_kwargs)
    else:
        model_kwargs.update({
            "entropy_coefficient": entropy_coefficient,
        })
        sac = QSAC(policy, action_value_model, action_value_twin_model, **model_kwargs)
    return sac
