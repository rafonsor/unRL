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


class Policy(pt.nn.Module):
    """Policy for discrete Action spaces"""
    def forward(self, state: pt.Tensor) -> pt.Tensor:
        """Returns logprobabilies of actions"""
        ...


class ContinuousPolicy(Policy):
    """Policy for continuous Action spaces"""
    def forward(self, state: pt.Tensor) -> pt.Tensor:
        """Returns a continuous-valued action"""
        ...


class GaussianPolicy(Policy):
    """Variational Policy of the Gaussian type"""
    def forward(self, state: pt.Tensor) -> t.Tuple[pt.Tensor, pt.Tensor]:
        """Returns the variational parameters, mu and sigma, of a Gaussian distribution from which to sample actions"""
        ...


class StateValueFunction(pt.nn.Module):
    """State-value function"""
    num_state_dims: int

    def forward(self, state: pt.Tensor) -> pt.Tensor:
        ...


class ActionValueFunction(pt.nn.Module):
    """Action-value function for discrete Action spaces"""
    num_state_dims: int
    num_actions: int

    def forward(self, state: pt.Tensor) -> pt.Tensor:
        """Returns action-value estimates for the entire Action space"""
        ...


class ContinuousActionValueFunction(pt.nn.Module):
    """Action-value function for continuous Action spaces"""
    num_state_dims: int
    num_action_dims: int

    def forward(self, state: pt.Tensor, action: pt.Tensor) -> pt.Tensor:
        """Returns action-value estimate for the requested action"""
        ...
