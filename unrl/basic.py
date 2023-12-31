# Copyright 2023 The unRL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math
import random

import torch as pt

import unrl.types as t


def argmax(seq: t.Sequence) -> int:
    max_ = -math.inf
    arg = -1
    for i, value in enumerate(seq):
        if value > max_:
            max_ = value
            arg = i
    return arg


def argmax_all(seq: t.Sequence) -> t.List[int]:
    max_ = -math.inf
    args = []
    for i, value in enumerate(seq):
        if value > max_:
            max_ = value
            args = [i]
        elif value == max_:
            args.append(i)
    return args


def argmax_random(seq: t.Sequence) -> int:
    if seq:
        return random.choice(argmax_all(seq))
    return -1


def sample_gaussian(mu: float, sigma: float) -> float:
    """Gaussian sampling using the Box-Muller transform"""
    u1, u2 = random.random(), random.random()
    z = math.sqrt(-2*math.log(u1)) * math.cos(2*math.pi*u2)
    return mu + z*sigma


def entropy(dist: t.Optional[pt.Tensor] = None, logits: t.Optional[pt.Tensor] = None) -> t.FloatLike:
    """Compute Entropy H for a probability distribution, either from probabilities or log-probabilities"""
    if dist is not None:
        dist /= dist.sum()
        return (-dist * dist.log()).sum()
    if logits is not None:
        return (-logits.exp() * logits).sum()
    return pt.Tensor([0])


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
    return reward + (1 - terminal) * discount_factor * successor_value.detach() - value


def mse(error: pt.Tensor) -> pt.Tensor:
    """Return the mean square of an error tensor"""
    return (error ** 2).mean()


def expected_value(values: pt.Tensor, logits: pt.Tensor) -> pt.Tensor:
    """Expected value for a discrete distribution"""
    return (logits.exp() * values).sum(dim=-1)


def rho_logits(target: pt.Tensor, behaviour: pt.Tensor, action: t.IntLike) -> pt.Tensor:
    """Calculate the importance sampling weight between target and behaviour log-probabilities for a given action"""
    return (target[action].exp() / behaviour[action].exp()).detach()


def rho_dists(target: pt.distributions.Distribution,
              behaviour: pt.distributions.Distribution,
              action: pt.Tensor) -> pt.Tensor:
    """Calculate the importance sampling weight between target and behaviour distributions for a given action"""
    return (target.log_prob(action).exp() / behaviour.log_prob(action).exp()).detach()
