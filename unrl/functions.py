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
        """Returns logprobabilities of actions"""
        ...


class ContinuousPolicy(Policy):
    """Policy for continuous Action spaces"""
    def forward(self, state: pt.Tensor) -> pt.Tensor:
        """Returns a continuous-valued action"""
        ...


class VariationalPolicy(Policy):
    """Variational Policy for continuous Action spaces"""
    def forward(self, state: pt.Tensor, *, dist: bool = False) -> t.Tuple | pt.distributions.Distribution:
        """Returns the variational parameters to use with `distribution` or a prebuilt distribution from which to sample
        continuous-valued actions"""
        ...

    @property
    def distribution(self) -> t.Type[pt.distributions.Distribution]:
        """Returns the distribution class to be parameterised by the Policy's outputs"""
        ...


class GaussianPolicy(VariationalPolicy):
    """Variational Policy of the Gaussian type"""
    def forward(self, state: pt.Tensor, *, dist: bool = False) -> t.Tuple[pt.Tensor, pt.Tensor] | pt.distributions.Normal:
        """Returns the variational parameters, mu and sigma, of a Gaussian distribution (or a prepared instance when
        `dist` is "True") from which to sample continuous-valued actions"""
        ...

    @property
    def distribution(self) -> t.Type[pt.distributions.Normal]:
        """Returns the distribution class to be parameterised by the Policy's outputs"""
        return pt.distributions.Normal


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


class DuelingActionValueFunction(pt.nn.Module):
    """Action-value function with two-headed outputs for discrete Action spaces"""
    num_state_dims: int
    num_actions: int

    def forward(self, state: pt.Tensor, *, combine: bool = True) -> pt.Tensor | t.Tuple[pt.Tensor, pt.Tensor]:
        """Returns action-value estimates for the entire Action space. Action-value estimates are composed from
        two-headed outputs state-value and action advantage estimates.

        Args:
            state: State
            combine: when "True", returns action-value estimates by directly summing state-value and action advantage
                     estimates.

        Returns:
            Either a tensor with action-value estimates, if `combine` is True, else a tuple (state-value, advantage).
        """
        ...


class DuelingContinuousActionValueFunction(pt.nn.Module):
    """Action-value function with two-headed outputs for continuous Action spaces

    Provides support for stochastic action-value estimates as described in [1]_.

    References:
        [1] Wang, Z., Bapst, V., Heess, N., & et al. (2016). "Sample efficient actor-critic with experience replay".
            arXiv:1611.01224. (Accepted as a poster in ICLR 2017.)
    """
    num_state_dims: int
    num_action_dims: int

    def forward(self,
                state: pt.Tensor,
                action: pt.Tensor,
                stochastic_actions: t.Optional[t.List[pt.Tensor]] = None,
                *,
                combine: bool = True
                ) -> pt.Tensor | t.Tuple[pt.Tensor, pt.Tensor, t.Optional[pt.Tensor]]:
        """Returns action-value estimates for the entire Action space. Action-value estimates are composed from
        two-headed outputs state-value and action advantage estimates.

        Args:
            state: State.
            action: action to provide an estimation for.
            stochastic_actions: (Optional) actions from which to compute an action-advantage expectation to subtract
                                from the value estimate.
            combine: when "True", returns action-value estimates by directly summing state-value and action advantage
                     estimates.

        Returns:
            Either a tensor with (stochastic) action-value estimates, when `combine` is True. Otherwise, a tuple (state-
            -value, advantage, advantage expectation). If no stochastic actions are supplied, the expectation is simply
            `None`.
        """
        ...


class NormalisedContinuousAdvantageFunction(DuelingContinuousActionValueFunction):
    """Action-value function for continuous Action spaces, with two-headed outputs parameterised by a state-based
    normalised advantage function (NAF, [1]_) that serves as a Policy.

    References:
        [1] Gu, S., Lillicrap, T., Sutskever, I., & et al. (2016). "Continuous deep q-learning with model-based
            acceleration". In Proceedings of The 33rd International Conference on Machine Learning.
    """

    def pick(self, state: pt.Tensor) -> pt.Tensor:
        """Returns the optimal action as determined by the underlying normalised advantage function parameterising
        action-value estimates.

        Args:
            state: State.

        Returns:
            A tensor with the greedy (continuous) action.
        """
        ...
