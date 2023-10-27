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
from unrl.algos.policy_gradient import ActorCritic, Policy, EligibilityTraceActorCritic
from unrl.algos.dqn import DQN, DQNExperienceReplay


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


class ExampleStateValueModel(pt.nn.Module):
    def __init__(self, num_state_dims, hidden_dim: int):
        super().__init__()
        self.layer1 = pt.nn.Linear(num_state_dims, hidden_dim)
        self.layer2 = pt.nn.Linear(hidden_dim, hidden_dim // 3)
        self.layer3 = pt.nn.Linear(hidden_dim // 3, 1)

    def forward(self, state: pt.Tensor) -> pt.Tensor:
        x = F.relu(self.layer1(state))
        x = F.relu(self.layer2(x))
        estimate = -F.relu(self.layer3(x))
        return estimate


class ExampleActionValueEstimator(pt.nn.Module):
    def __init__(self, num_state_dims: int, num_actions: int, hidden_dim: int):
        super().__init__()
        self.layer1 = pt.nn.Linear(num_state_dims, hidden_dim)
        self.layer2 = pt.nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = pt.nn.Linear(hidden_dim, num_actions)

    def forward(self, state: pt.Tensor) -> pt.Tensor:
        x = F.relu(self.layer1(state))
        x = F.relu(self.layer2(x))
        return -F.relu(self.layer3(x))


def prepare_game_model_actorcritic(num_state_dims: int,
                                   num_actions: int,
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
    policy = ExamplePolicy(num_state_dims, num_actions, hidden_dim_policy)
    state_value_model = ExampleStateValueModel(num_state_dims, hidden_dim_values)
    action_sampler = make_sampler(ActionSamplingMode.EPSILON_GREEDY)
    if eligibility_traces:
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


def prepare_game_model_dqn(num_state_dims: int, num_actions: int, buffer_size: t.Optional[int] = None) -> DQN:
    discount_factor = 0.99
    learning_rate = 1e-4
    epsilon_greedy = 0.1
    refresh_steps = 250
    hidden_dim = 10
    replay_minibatch = 32
    action_value_model = ExampleActionValueEstimator(num_state_dims, num_actions, hidden_dim)
    if buffer_size:
        return DQNExperienceReplay(
            action_value_model,
            learning_rate=learning_rate, discount_factor=discount_factor, epsilon_greedy=epsilon_greedy,
            target_refresh_steps=refresh_steps, replay_memory_capacity=buffer_size, batch_size=replay_minibatch)
    else:
        return DQN(action_value_model, learning_rate=learning_rate, discount_factor=discount_factor,
                   epsilon_greedy=epsilon_greedy, target_refresh_steps=refresh_steps)