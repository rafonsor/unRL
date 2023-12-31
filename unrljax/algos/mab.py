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
"""This Module provides Multi-Armed Bandits for Discrete Action Spaces in (non)stationary settings."""
from __future__ import annotations

from abc import abstractmethod, ABCMeta
from enum import Enum
from functools import wraps

import jax.numpy as jnp
from jax import random

import unrljax.types as t
from unrljax.basic import argmax_random, logsoftmax, softmax
from unrljax.mops import transpose_last2
from unrljax.random import PRNGMixin


__all__ = [
    "MultiArmedBanditConfig",
    "MultiArmedBanditProtocol",
    # Implementations
    "GradientMultiArmedBandit",
    "IncrementalMultiArmedBandit",
    "IncrementalMultiArmedBanditEpsilonGreedy",
    "IncrementalMultiArmedBanditUCB",
    "LinUCB",
    "LinUCBHybrid",
    "MultiArmedBandit",
    "MultiArmedBanditEpsilonGreedy",
    "MultiArmedBanditThompson",
    "MultiArmedBanditThompsonDynamic",
]


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


class MultiArmedBanditProtocol(t.Protocol):
    
    __init__: t.Callable[(MultiArmedBanditConfig, ...), MultiArmedBanditProtocol]
    """Signature: `__init__(config: MultiArmedBanditConfig, *args, **kwargs)"""
    
    pick: t.Callable[..., int]
    """Signature: `pick(self, *args, **kwargs) -> int`
    Pick and return one action"""

    update: t.Callable[(int, float, ...), None]
    """Signature: `update(self, action: int, reward: float, *args, **kwargs) -> None`
    Update the bandit corresponding to the chosen action using the observed reward"""


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
    """Multi-armed Bandit for discrete Action space

    References:
    [1] Sutton, R. S., & Barto, A. G. (2018). Section 2.2., "Reinforcement learning: An introduction (2nd ed.)". The MIT
        Press.
    """
    counts: t.IntArray
    estimates: t.FloatArray
    rewards: t.List[t.List[float]]
    never_chosen: t.Set[int]

    def set_state(self, counts: t.IntArray, estimates: t.FloatArray, rewards: t.List[t.List[float]]):
        """Set the MAB model to a specific state

        Args:
            counts: Vector of visit counts for each bandit.
            estimates: Vector of State-value estimates for each bandit.
            rewards: List of rewards received from each bandit.
        """
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
    """Incremental Sample-average Multi-armed Bandit for discrete Action space

    References:
    [1] Sutton, R. S., & Barto, A. G. (2018). Section 2.4-5., Reinforcement learning: An introduction (2nd ed.). The MIT
        Press.
    """
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
    """Incremental Sample-average Multi-armed Bandit for discrete Action space with Upper-Confidence Bound sampling

    Reference:
    [1] Sutton, R. S., & Barto, A. G. (2018). Section 2.7., Reinforcement learning: An introduction (2nd ed.). The MIT
        Press.
    """
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


class MultiArmedBanditThompson(MultiArmedBanditBase):
    """Multi-armed Bandit for discrete Action space with Thompson sampling"""
    counts: t.IntArray
    estimates: t.FloatArray

    POSITIVE = 0
    NEGATIVE = 1

    def set_state(self, counts: t.IntArray, estimates: t.FloatArray):
        assert len(counts) == self.config.k
        assert all(c >= 0 for c in counts)
        assert len(estimates) == self.config.k
        self.counts = counts.copy()
        self.estimates = estimates.copy()

    def reset_state(self):
        # Two-dimensional, tracks occurrences of "positive" and "negative" rewards. Note, we initialise counts to 1 to
        # remove need for increment when sampling from their beta distributions.
        self.counts = jnp.ones((2, self.config.k,))
        self.estimates = self.config.initial_q_estimates.copy()

    def pick(self) -> int:
        theta = random.beta(self.prng_key(), self.counts[self.POSITIVE], self.counts[self.NEGATIVE])
        return argmax_random(self.prng_key(), theta)

    def update(self, action: int, reward: float):
        if action < 0 or action >= self.config.k:
            raise IndexError("Unsupported action")
        kind = self.POSITIVE if reward > self.config.thompson_reward_threshold else self.NEGATIVE
        self.counts = self.counts.at[kind, action].add(1)
        self.estimates = self.estimates.at[action].add((reward - self.estimates[action]) / self.counts[:, action].sum())


class MultiArmedBanditThompsonDynamic(MultiArmedBanditThompson):
    """Multi-armed Bandit for discrete Action space with Thompson sampling with dynamic reward binarisation using
    current estimates"""
    estimates: t.FloatArray

    def update(self, action: int, reward: float):
        if action < 0 or action >= self.config.k:
            raise IndexError("Unsupported action")
        kind = self.POSITIVE if reward > self.estimates[action] else self.NEGATIVE
        self.counts = self.counts.at[kind, action].add(1)
        self.estimates = self.estimates.at[action].add((reward - self.estimates[action]) / self.counts[:, action].sum())


class GradientMultiArmedBandit(MultiArmedBanditBase):
    """Gradient-based Multi-armed Bandit for discrete Action space

    Reference:
    [1] Sutton, R. S., & Barto, A. G. (2018). Section 2.8., Reinforcement learning: An introduction (2nd ed.). The MIT
        Press.
    """
    steps: int
    preferences: t.FloatArray
    _latest_probs: t.FloatArray
    _average_reward: float

    def set_state(self, preferences: t.FloatArray):
        assert len(preferences) == self.config.k
        self.preferences = preferences.copy()
        self._latest_probs = softmax(self.preferences)
        self.steps = 0
        self._average_reward = 0

    def reset_state(self):
        self.preferences = jnp.zeros((self.config.k,), dtype=jnp.float32)
        self._latest_probs = jnp.ones((self.config.k,), dtype=jnp.float32) / self.config.k
        self.steps = 0
        self._average_reward = 0

    def pick(self) -> int:
        logits = logsoftmax(self.preferences)
        self._latest_probs = softmax(logits)
        return random.categorical(self.prng_key(), logits=logits).item()

    def update(self, action: int, reward: float):
        self.steps += 1
        # Compute new average reward `\bar{R}_t`
        r_delta_tm1 = reward - self._average_reward
        self._average_reward += r_delta_tm1 / self.steps
        # Compute reward "error" at t
        r_delta_t = reward - self._average_reward

        p_delta = -1 * r_delta_t * self._latest_probs  # Descend on actions not chosen
        p_delta = p_delta.at[action].set(r_delta_t * (1 - self._latest_probs[action]))  # Ascend on chosen action
        self.preferences += p_delta


class LinUCB(PRNGMixin, MultiArmedBanditProtocol):
    """Contextual Upper-Confidence Bound Multi-Armed Bandit with linear payfoff using disjoint linear models.

    Reference:
    [1] Li, L., Chu, W., Langford, J., & Schapire, R. E. (2010). A contextual-bandit approach to personalized news
        article recommendation. In Proceedings of the 19th international conference on World wide web.
    """
    config: MultiArmedBanditConfig
    weights: t.FloatArray  # (k x d x d)
    biases: t.FloatArray  # (k x d x 1)

    _latent_dim: int
    
    prng_collection = 'LinUCB'

    def __init__(self, config: MultiArmedBanditConfig, latent_dim: int):
        super().__init__(self.prng_collection)
        self.config = config
        self._latent_dim = latent_dim
        self.reset_state()

    def reset_state(self):
        self.weights = jnp.zeros((self.config.k, self._latent_dim, self._latent_dim), dtype=jnp.float32)
        self.weights += jnp.identity(self._latent_dim, dtype=jnp.float32)
        self.biases = jnp.zeros((self.config.k, self._latent_dim, 1), dtype=jnp.float32)

    @property
    def estimates(self) -> t.FloatArray:
        unit_features = jnp.ones((self.config.k, self._latent_dim, 1))
        return self._compute_contextual_estimates(self.weights, self.biases, unit_features)

    @staticmethod
    def _compute_contextual_estimates(weights: t.Array, biases: t.Array, features: t.Array, invert_weights: bool = True) -> t.Array:
        if invert_weights:
            weights = jnp.linalg.inv(weights)
        theta = jnp.einsum('kdd,kd1->kd1', weights, biases)
        return jnp.einsum('k1d,kd1->k', transpose_last2(theta), features)

    def check_and_prepare_features(self, features: t.Array):
        assert features.shape[-2:] == (self.config.k, self._latent_dim), \
            f"Features dimension mismatch: expecting (..., {self.config.k}, {self._latent_dim}), not {features.shape}."
        return jnp.expand_dims(features, -1)

    def pick(self, features: t.Array) -> int:
        features = self.check_and_prepare_features(features)
        # Generalises single-action form of Algorithm 1 ref.[1].
        inv_weights = jnp.linalg.inv(self.weights)
        p1 = self._compute_contextual_estimates(inv_weights, self.biases, features, invert_weights=False)
        p2 = jnp.einsum('k1d,kdd,kd1->k', transpose_last2(features), inv_weights, features)
        p = p1 + self.config.ucb_exploration * jnp.sqrt(p2)
        return argmax_random(self.prng_key(), p)

    def update(self, action: int, reward: int, *, features: t.Array):
        features = self.check_and_prepare_features(features)
        x = features[action, :]
        self.weights = self.weights.at[action].add(x @ transpose_last2(x))
        self.biases = self.biases.at[action].add(reward * x)


class LinUCBHybrid(PRNGMixin, MultiArmedBanditProtocol):
    """Contextual Upper-Confidence Bound Multi-Armed Bandit with linear payfoff using hybrid linear models.

    Reference:
    [1] Li, L., Chu, W., Langford, J., & Schapire, R. E. (2010). A contextual-bandit approach to personalized news
        article recommendation. In Proceedings of the 19th international conference on World wide web.
    """
    config: MultiArmedBanditConfig
    A0: t.FloatArray  # (k x k)
    b0: t.FloatArray  # (k x 1)
    A: t.FloatArray  # (k x d x d)
    B: t.FloatArray  # (k x d x k)
    biases: t.FloatArray  # (k x d x 1)

    _latent_dim: int

    prng_collection = 'LinUCBHybrid'

    def __init__(self, config: MultiArmedBanditConfig, latent_dim: int):
        super().__init__(self.prng_collection)
        self.config = config
        self._latent_dim = latent_dim
        self.reset_state()

    def reset_state(self):
        self.A0 = jnp.identity(self.config.k, dtype=jnp.float32)
        self.b0 = jnp.zeros((self.config.k, 1), dtype=jnp.float32)
        self.A = jnp.zeros((self.config.k, self._latent_dim, self._latent_dim), dtype=jnp.float32) + jnp.identity(
            self._latent_dim, dtype=jnp.float32)
        self.B = jnp.zeros((self.config.k, self._latent_dim, self.config.k), dtype=jnp.float32)
        self.biases = jnp.zeros((self.config.k, self._latent_dim, 1), dtype=jnp.float32)

    @property
    def estimates(self) -> t.FloatArray:
        unit_features = jnp.ones((self.config.k, self.config.k + self._latent_dim, 1))
        zt = transpose_last2(unit_features[:, :self.config.k, :])
        xt = transpose_last2(unit_features[:, self.config.k:, :])
        _, _, beta, theta = self._invert_and_compute_beta_theta()
        return zt @ beta + jnp.einsum('k1d,kd1->k', xt, theta)

    def check_and_prepare_features(self, features: t.Array) -> t.Tuple[t.Array, t.Array]:
        expected_dims = (self.config.k, self.config.k + self._latent_dim)
        assert features.shape[-2:] == expected_dims, \
            f"Features dimension mismatch: expecting (..., {expected_dims}), not {features.shape}."
        features = jnp.expand_dims(features, -1)
        return features[:, :self.config.k, :], features[:, self.config.k:, :]

    def _invert_and_compute_beta_theta(self) -> t.Tuple[t.FloatArray, t.FloatArray, t.FloatArray, t.FloatArray]:
        A0_inv = jnp.linalg.inv(self.A0)  # kk
        A_inv = jnp.linalg.inv(self.A)  # kdd
        beta = A0_inv @ self.b0  # k1
        theta = jnp.einsum('kdd,kd1->kd1', A_inv, self.biases - self.B @ beta)  # kdd . (kd1 - (kdk . k1)) = kd1
        return A0_inv, A_inv, beta, theta

    def pick(self, features: t.Array) -> int:
        (z, x) = self.check_and_prepare_features(features)  # kk1, kd1

        zt, xt = transpose_last2(z), transpose_last2(x)  # k1k, k1d
        A0_inv, A_inv, beta, theta = self._invert_and_compute_beta_theta()

        t1 = xt @ A_inv  # k1d . kdd = k1d
        t2 = A0_inv @ transpose_last2(self.B) @ A_inv @ x  # kk . kkd . kdd . kd1 = kk1

        s1 = jnp.einsum('k1k,kk,kk1->k', zt, A0_inv, z)
        s2 = jnp.einsum('k1k,kk1->k', 2*zt, t2)
        s3 = jnp.einsum('k1d,kd1->k', t1, x)
        s4 = jnp.einsum('k1d,kdk,kk1->k', t1, self.B, t2)
        s = s1 - s2 + s3 + s4  # k

        p1 = jnp.einsum('k1k,k1->k', zt, beta)
        p2 = jnp.einsum('k1d,kd1->k', xt, theta)
        p3 = self.config.ucb_exploration * jnp.sqrt(s)
        p = p1 + p2 + p3  # k

        return argmax_random(self.prng_key(), p)

    def update(self, action: int, reward: int, *, features: t.Array):
        (z, x) = self.check_and_prepare_features(features)  # kk1, kd1

        za, xa = z[action], x[action]
        BA_inv = self.B[action].T @ jnp.linalg.inv(self.A[action])

        self.A0 += BA_inv @ self.B[action] + za @ za.T
        self.b0 += BA_inv @ self.biases[action] + reward * za
        self.A = self.A.at[action].add(xa @ xa.T)
        self.B = self.B.at[action].add(xa @ za.T)
        self.biases = self.biases.at[action].add(reward * xa)

        BA_inv_new = self.B[action].T @ jnp.linalg.inv(self.A[action])
        self.A0 -= BA_inv_new @ self.B[action]
        self.b0 -= BA_inv_new @ self.biases[action]
