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
"""Module provides sampling strategies for selecting actions using their log probabilities"""
from enum import Enum

import torch as pt

import unrl.types as t
from unrl.config import validate_config

__all__ = [
    "ActionSampler",
    "ActionSamplingMode",
    "make_sampler",
    # Sampling strategies
    "GreedyActionSampler",
    "EpsilonGreedyActionSampler",
    "StochasticActionSampler",
    # Functional API
    "greedy_action_sampler",
    "epsilon_greedy_action_sampler",
    "stochastic_action_sampler",
]

DEFAULT_EPSILON = 0.1


def greedy_action_sampler(logits: pt.Tensor) -> pt.Tensor:
    return pt.argmax(logits, dim=-1)


def epsilon_greedy_action_sampler(logits: pt.Tensor, epsilon: float) -> pt.Tensor:
    if pt.rand(1).item() <= epsilon:
        if len(logits.shape) == 1:
            logits = logits[None, ...]
        return pt.randint(0, logits.shape[-1], (logits.shape[0],))
    return greedy_action_sampler(logits)


def stochastic_action_sampler(logits: pt.Tensor) -> pt.Tensor:
    return pt.distributions.Categorical(logits=logits).sample()


class ActionSampler(t.Protocol):
    def sample(self, logits: pt.Tensor) -> pt.Tensor:
        ...


class GreedyActionSampler(ActionSampler):
    """Always select the best-valued action"""
    def sample(self, logits: pt.Tensor) -> pt.Tensor:
        return greedy_action_sampler(logits)


class EpsilonGreedyActionSampler(ActionSampler):
    """Select with probability `epsilon` one action uniformly at random, otherwise choose best-valued action (Greedy)"""
    def __init__(self, epsilon: float = DEFAULT_EPSILON):
        validate_config(epsilon, 'epsilon', 'unitpositive')
        self.epsilon = epsilon

    def sample(self, logits: pt.Tensor) -> pt.Tensor:
        return epsilon_greedy_action_sampler(logits, self.epsilon)


class StochasticActionSampler(ActionSampler):
    """Sample action according to a probability distribution"""
    def sample(self, logits: pt.Tensor) -> pt.Tensor:
        return stochastic_action_sampler(logits)


class ActionSamplingMode(Enum):
    GREEDY = 'greedy'  # Always select the best-value actions, breaking ties randomly with equal probability
    EPSILON_GREEDY = 'epsilon'  # Select with probability `epsilon` one action uniformly at random, otherwise choose best-valued action
    STOCHASTIC = 'stochastic'  # Sample action according to distribution


def make_sampler(mode: ActionSamplingMode, **kwargs) -> ActionSampler:
    if mode == ActionSamplingMode.GREEDY:
        return GreedyActionSampler()
    if mode == ActionSamplingMode.EPSILON_GREEDY:
        return EpsilonGreedyActionSampler(**kwargs)
    if mode == ActionSamplingMode.STOCHASTIC:
        return StochasticActionSampler()
    raise NotImplementedError(f"Unsupported action sampling strategy: `{mode}`.")
