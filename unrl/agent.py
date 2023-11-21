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
import unrl.types as t

import torch as pt

from unrl.functions import ActionValueFunction, DuelingActionValueFunction, Policy, ContinuousPolicy, VariationalPolicy


class Agent(t.Protocol):
    def pick(self, state: pt.Tensor) -> pt.Tensor:
        """Pick greedy action for the current state"""
        ...


class QAgent(Agent):
    """Agent driven greedily by an optimal Action-value Function for discrete Action spaces"""
    def __init__(self, action_value_model: ActionValueFunction | DuelingActionValueFunction):
        self.action_value_model = action_value_model

    def pick(self, state: pt.Tensor) -> pt.Tensor:
        action_values = self.action_value_model(state)
        return pt.argmax(action_values, dim=-1)


class PolicyAgent(Agent):
    """Optimal Policy Agent for discrete Action spaces"""
    def __init__(self, policy: Policy):
        self.policy = policy

    def pick(self, state: pt.Tensor) -> pt.Tensor:
        logprobs = self.policy(state)
        return pt.argmax(logprobs, dim=-1)


class ContinuousPolicyAgent(Agent):
    """Optimal Policy Agent for continuous Action spaces"""
    def __init__(self, policy: ContinuousPolicy):
        self.policy = policy

    def pick(self, state: pt.Tensor) -> pt.Tensor:
        return self.policy(state)


class VariationalPolicyAgent(Agent):
    """Agent driven greedily by a probabilistic Policy for continuous Action spaces"""
    def __init__(self, policy: VariationalPolicy):
        self.policy = policy

    def pick(self, state: pt.Tensor) -> pt.Tensor:
        dist = self.policy.forward(state, dist=True)
        return dist.mode
