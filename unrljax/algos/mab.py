"""This Module provides Multi-Armed Bandits for Discrete Action Spaces in (non)stationary settings."""
from __future__ import annotations

from abc import abstractmethod, ABCMeta
from enum import Enum
from functools import wraps

from jax import numpy as jnp
from jax import random

import unrljax.types as t
from unrljax.basic import argmax_random, logsoftmax, softmax
from unrljax.random import PRNGMixin


class MultiArmedBanditProtocol(t.Protocol):

    def pick(self, *args, **kwargs) -> int:
        ...

    def update(self, action: int, reward: float) -> None:
        ...


def epsilon_greedy(cls: t.Type[MultiArmedBanditProtocol]) -> t.Type:
    orig__init__ = cls.__init__
    origpick = cls.pick

    @wraps(cls.__init__)
    def __init__(self, *args, epsilon: float, **kwargs):
        orig__init__(self, *args, **kwargs)
        assert 0 <= epsilon <= 1, "Epsilon-greedy requires an epsilon probability"
        self.epsilon = epsilon

    @wraps(cls.pick)
    def pick(self, *args, **kwargs) -> int:
        if random.normal(self.prng_key()) > self.epsilon:
            return origpick(self, *args, **kwargs)
        return random.randint(self.prng_key(), (1, ), 0, self.config.k - 1).item()

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
    initial_q_estimates: t.FloatArray = 0.0
    ucb_exploration: float = None
    thompson_reward_threshold: float = 0.5

    prng_collection: str = 'MultiArmedBandit'
    
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

        if 'thompson_reward_threshold' in kwargs:
            self.thompson_reward_threshold = kwargs.pop('thompson_reward_threshold')

        if isinstance(self.initial_q_estimates, float):
            self.initial_q_estimates = jnp.ones((self.k,), dtype=jnp.float32) * self.initial_q_estimates
        else:
            self.initial_q_estimates = jnp.array(self.initial_q_estimates, dtype=jnp.float32)

    def validate_config(self):
        assert self.k > 1, "MAB requires multiple actions to be available"

        if self.average_mode == self.AverageMode.WEIGHTED:
            assert self.weighted_average_alpha, "Weighted average requires a `weighted_average_alpha` step size"

        if self.weighted_average_alpha:
            assert 0 <= self.weighted_average_alpha < 1, "Weighted average step size must belong to unit interval"

        assert isinstance(self.initial_q_estimates, t.FloatArray) and self.initial_q_estimates.dtype == jnp.float32, \
            "Initial action-value estimates must be numeric"

        if self.ucb_exploration:
            assert self.ucb_exploration >= 0, "Upper-Confidence Bound exploration rate must be non-negative"


class MultiArmedBanditBase(PRNGMixin, MultiArmedBanditProtocol, metaclass=ABCMeta):
    config: MultiArmedBanditConfig

    def __init__(self, config: MultiArmedBanditConfig):
        super().__init__(config.prng_collection)
        self.config = config
        self.reset_state()

    @abstractmethod
    def set_state(self, *args, **kwargs) -> int:
        ...

    @abstractmethod
    def reset_state(self):
        ...

    @abstractmethod
    def pick(self, *args, **kwargs) -> int:
        ...

    @abstractmethod
    def update(self, action: int, reward: float) -> None:
        ...


class MultiArmedBandit(MultiArmedBanditBase):
    """Multi-armed Bandit for discrete Action space"""
    counts: t.IntArray
    estimates: t.FloatArray
    rewards: t.List[t.List[float]]
    never_chosen: t.Set[int]

    def set_state(self, counts: t.IntArray, estimates: t.FloatArray, rewards: t.List[t.List[float]]):
        assert len(counts) == self.config.k
        assert all(c >= 0 for c in counts)
        assert len(estimates) == self.config.k
        assert all(len(history) == self.config.k for history in rewards)
        self.counts = counts
        self.estimates = estimates
        self.rewards = rewards
        self.never_chosen = {action for action in range(self.config.k) if self.counts[action] == 0}

    def reset_state(self):
        self.counts = jnp.zeros(self.config.k, dtype=int)
        self.estimates = jnp.array(self.config.initial_q_estimates, dtype=float)
        self.rewards = [[]] * self.config.k
        self.never_chosen = set(range(self.config.k))

    def pick(self) -> int:
        return jnp.argmax(self.estimates).item()

    def update(self, action: int, reward: float):
        if action < 0 or action >= self.config.k:
            raise IndexError("Unsupported action")
        # Thread unsafe
        self.counts = self.counts.at[action].add(1)
        self.rewards[action].append(reward)

        if self.config.average_mode == self.config.AverageMode.SAMPLE:
            q = sum(self.rewards[action]) / self.counts[action]
        elif self.config.average_mode == self.config.AverageMode.WEIGHTED:
            q = self._exponentially_weighted_reward_sum(action, self.counts[action].item())
        else:
            return

        self.estimates = self.estimates.at[action].set(q)

    def _exponentially_weighted_reward_sum(self, action: int, n: int) -> float:
        x = jnp.power(1 - self.config.weighted_average_alpha, n) * self.config.initial_q_estimates[action]
        for i, reward in enumerate(self.rewards[action]):
            x += self.config.weighted_average_alpha * jnp.power(1 - self.config.weighted_average_alpha, n - i) * reward
        return x.item()


@epsilon_greedy
class MultiArmedBanditEpsilonGreedy(MultiArmedBandit):
    """Epsilon-greedy Multi-armed Bandit for discrete Action space"""


class IncrementalMultiArmedBandit(MultiArmedBanditBase):
    """Incremental Sample-average Multi-armed Bandit for discrete Action space"""
    counts: t.IntArray
    estimates: t.FloatArray
    never_chosen: t.Set[int]

    def set_state(self, counts: t.IntArray, estimates: t.FloatArray):
        assert len(counts) == self.config.k
        assert all(c >= 0 for c in counts)
        assert len(estimates) == self.config.k
        self.counts = counts
        self.estimates = estimates
        self.never_chosen = {action for action in range(self.config.k) if self.counts[action] == 0}

    def reset_state(self):
        self.counts = jnp.zeros((self.config.k,))
        self.estimates = self.config.initial_q_estimates.copy()
        self.never_chosen = set(range(self.config.k))

    pick = MultiArmedBandit.pick

    def update(self, action: int, reward: float):
        if action < 0 or action >= self.config.k:
            raise IndexError("Unsupported action")
        # Thread unsafe
        self.counts = self.counts.at[action].add(1)
        self.estimates = self.estimates.at[action].add((reward - self.estimates[action]) / self.counts[action])


@epsilon_greedy
class IncrementalMultiArmedBanditEpsilonGreedy(IncrementalMultiArmedBandit):
    """Incremental Sample-average Epsilon-greedy Multi-armed Bandit for discrete Action space"""


class IncrementalMultiArmedBanditUCB(MultiArmedBanditBase):
    """Incremental Sample-average Multi-armed Bandit for discrete Action space with Upper-Confidence Bound sampling"""
    steps: int
    counts: t.IntArray
    estimates: t.FloatArray
    never_chosen: t.List[int]

    def set_state(self, counts: t.IntArray, estimates: t.FloatArray):
        assert len(counts) == self.config.k
        assert all(c >= 0 for c in counts)
        assert len(estimates) == self.config.k
        self.counts = counts.copy()
        self.steps = sum(counts)
        self.estimates = estimates.copy()
        self.never_chosen = [action for action in range(self.config.k) if self.counts[action] == 0]

    def reset_state(self):
        self.counts = jnp.zeros((self.config.k,))
        self.steps = 0
        self.estimates = self.config.initial_q_estimates.copy()
        self.never_chosen = list(range(self.config.k))

    def pick(self) -> int:
        if self.never_chosen:
            return random.choice(self.prng_key(), jnp.array(self.never_chosen)).item()
        ucb = self.estimates + self.config.ucb_exploration * jnp.sqrt(jnp.log(self.steps) / self.counts)
        return argmax_random(self.prng_key(), ucb)

    def update(self, action: int, reward: float):
        if action < 0 or action >= self.config.k:
            raise IndexError("Unsupported action")
        if action in self.never_chosen:
            self.never_chosen.remove(action)
        # Thread unsafe
        self.steps += 1
        self.counts = self.counts.at[action].add(1)
        self.estimates = self.estimates.at[action].add((reward - self.estimates[action]) / self.counts[action])
