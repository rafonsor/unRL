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
import typing as t
from enum import Enum
from functools import wraps

from unrl.basic import argmax_random


def epsilon_greedy(cls: t.Type) -> t.Type:
    orig__init__ = cls.__init__
    origpick = cls.pick

    @wraps(cls.__init__)
    def __init__(self, *args, epsilon: float, **kwargs):
        orig__init__(self, *args, **kwargs)
        assert 0 <= epsilon <= 1, "Epsilon-greedy requires an epsilon probability"
        self.epsilon = epsilon

    @wraps(cls.pick)
    def pick(self, *args, **kwargs) -> int:
        if random.random() > self.epsilon:
            return origpick(self, *args, **kwargs)
        return random.randint(0, self.config.k - 1)

    cls.__init__ = __init__
    cls.pick = pick
    return cls


class MultiArmedBanditConfig:
    class AverageMode(Enum):
        SAMPLE = 'sample'
        WEIGHTED = 'weighted'

    k: int
    average_mode: AverageMode = AverageMode.SAMPLE
    weighted_average_alpha: float = None
    initial_q_estimates: float | t.List[float] = 0.0
    ucb_exploration: float = None
    
    def __init__(self, k: int, **kwargs):
        self.update_config(k, **kwargs)
        self.validate_config()

    def update_config(self, k: int, **kwargs):
        self.k = k

        if 'average_mode' in kwargs:
            average_mode = kwargs.pop('average_mode')
            if isinstance(average_mode, self.AverageMode):
                self.average_mode = average_mode
            else:
                self.average_mode = self.AverageMode(average_mode)

        if 'weighted_average_alpha' in kwargs:
            weighted_average_alpha = kwargs.pop('weighted_average_alpha')
            assert 0 <= weighted_average_alpha < 1, "Weighted average step size must belong to unit interval"

        if 'initial_q_estimates' in kwargs:
            self.initial_q_estimates = kwargs.pop('initial_q_estimates')

        if 'ucb_exploration' in kwargs:
            self.ucb_exploration = kwargs.pop('ucb_exploration')

        if isinstance(self.initial_q_estimates, float):
            self.initial_q_estimates = [self.initial_q_estimates] * self.k

    def validate_config(self):
        assert self.k > 1, "MAB requires multiple actions to be available"

        if self.average_mode == self.AverageMode.WEIGHTED:
            assert self.weighted_average_alpha, "Weighted average requires a `weighted_average_alpha` step size"

        if self.weighted_average_alpha:
            assert 0 <= self.weighted_average_alpha < 1, "Weighted average step size must belong to unit interval"

        assert isinstance(self.initial_q_estimates, list) and all(isinstance(q, float) for q in self.initial_q_estimates), \
            "Initial action-value estimates must be numeric"

        if self.ucb_exploration:
            assert self.ucb_exploration >= 0, "Upper-Confidence Bound exploration rate must be non-negative"


class MultiArmedBandit:
    """Multi-armed Bandit for discrete Action space"""
    config: MultiArmedBanditConfig
    counts: t.List[int]
    estimates: t.List[float]
    rewards: t.List[t.List[float]]
    never_chosen: t.List[int]

    def __init__(self, config: MultiArmedBanditConfig):
        self.config = config
        self.reset_state()

    def set_state(self, counts: t.List[int], estimates: t.List[float], rewards: t.List[t.List[float]]):
        assert len(counts) == self.config.k
        assert all(c >= 0 for c in counts)
        assert len(estimates) == self.config.k
        assert all(len(history) == self.config.k for history in rewards)
        self.counts = counts
        self.estimates = estimates
        self.rewards = rewards
        self.never_chosen = [action for action in range(self.config.k) if self.counts[action] == 0]

    def reset_state(self):
        self.counts = [0] * self.config.k
        self.estimates = self.config.initial_q_estimates.copy()
        self.rewards = [[]] * self.config.k
        self.never_chosen = list(range(self.config.k))

    def pick(self) -> int:
        return argmax_random(self.estimates)

    def update(self, action: int, reward: float):
        if action < 0 or action >= self.config.k:
            raise IndexError("Unsupported action")
        # Thread unsafe
        n = self.counts[action] + 1
        self.counts[action] = n
        self.rewards[action].append(reward)

        if self.config.average_mode == self.config.AverageMode.SAMPLE:
            self.estimates[action] = sum(self.rewards[action]) / n
        elif self.config.average_mode == self.config.AverageMode.WEIGHTED:
            self.estimates[action] = self._exponentially_weighted_reward_sum(action, n)

    def _exponentially_weighted_reward_sum(self, action: int, n: int) -> float:
        return sum(
            self.config.weighted_average_alpha * (1 - self.config.weighted_average_alpha) ** (n - i) * reward
            for i, reward in enumerate(self.rewards[action])
        ) + (1 - self.config.weighted_average_alpha) ** n * self.config.initial_q_estimates[action]


@epsilon_greedy
class MultiArmedBanditEpsilonGreedy(MultiArmedBandit):
    """Epsilon-greedy Multi-armed Bandit for discrete Action space"""


class IncrementalMultiArmedBandit:
    """Incremental Sample-average Multi-armed Bandit for discrete Action space"""
    config: MultiArmedBanditConfig
    counts: t.List[int]
    estimates: t.List[float]
    never_chosen: t.List[int]

    def __init__(self, config: MultiArmedBanditConfig):
        assert config.average_mode == config.AverageMode.SAMPLE, "Only sample-average is supported in incremental mode"

        self.config = config
        self.reset_state()

    def set_state(self, counts: t.List[int], estimates: t.List[float]):
        assert len(counts) == self.config.k
        assert all(c >= 0 for c in counts)
        assert len(estimates) == self.config.k
        self.counts = counts
        self.estimates = estimates
        self.never_chosen = [action for action in range(self.config.k) if self.counts[action] == 0]

    def reset_state(self):
        self.counts = [0] * self.config.k
        self.estimates = self.config.initial_q_estimates.copy()
        self.never_chosen = list(range(self.config.k))

    pick = MultiArmedBandit.pick

    def update(self, action: int, reward: float):
        if action < 0 or action >= self.config.k:
            raise IndexError("Unsupported action")
        # Thread unsafe
        n = self.counts[action] + 1
        self.counts[action] = n
        self.estimates[action] += (reward - self.estimates[action]) / n


@epsilon_greedy
class IncrementalMultiArmedBanditEpsilonGreedy(IncrementalMultiArmedBandit):
    """Incremental Sample-average Epsilon-greedy Multi-armed Bandit for discrete Action space"""


class IncrementalMultiArmedBanditUCB:
    """Incremental Sample-average Multi-armed Bandit for discrete Action space with Upper-Confidence Bound sampling"""
    config: MultiArmedBanditConfig
    counts: t.List[int]
    steps: int
    estimates: t.List[float]
    never_chosen: t.List[int]

    def __init__(self, config: MultiArmedBanditConfig):
        assert config.average_mode == config.AverageMode.SAMPLE, "Only sample-average is supported in incremental mode"

        self.config = config
        self.reset_state()

    def set_state(self, counts: t.List[int], estimates: t.List[float]):
        assert len(counts) == self.config.k
        assert all(c >= 0 for c in counts)
        assert len(estimates) == self.config.k
        self.counts = counts
        self.steps = sum(counts)
        self.estimates = estimates
        self.never_chosen = [action for action in range(self.config.k) if self.counts[action] == 0]

    def reset_state(self):
        self.counts = [0] * self.config.k
        self.steps = 0
        self.estimates = self.config.initial_q_estimates.copy()
        self.never_chosen = list(range(self.config.k))

    def pick(self) -> int:
        if self.never_chosen:
            return random.choice(self.never_chosen)
        return argmax_random(
            q + self.config.ucb_exploration * math.sqrt(math.log(self.steps) / self.counts[action])
            for action, q in enumerate(self.estimates)
        )

    def update(self, action: int, reward: float):
        if action < 0 or action >= self.config.k:
            raise IndexError("Unsupported action")

        if action in self.never_chosen:
            self.never_chosen.remove(action)

        # Thread unsafe
        self.steps += 1
        n = self.counts[action] + 1
        self.counts[action] = n
        self.estimates[action] += (reward - self.estimates[action]) / n