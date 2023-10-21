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
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
from jax import random

from unrljax.algos.mab import *
from unrljax.basic import sample_gaussian
from unrljax.random import get_prng_key
from unrljax.types import FloatArray


class GaussianBandits:
    prng_collection: str = 'GaussianBandits'

    def __init__(self,
                 k: int,
                 epsilon: float = None,
                 confidence_bound: float = None,
                 thompson_sampling: bool = False,
                 mus: FloatArray = None,
                 sigmas: FloatArray = None):
        self.mus = jnp.zeros((k,), dtype=jnp.float32) if mus is None else mus
        self.sigmas = jnp.ones((k,), dtype=jnp.float32) if sigmas is None else sigmas

        print("Best action and value:", self.mus.argmax(), self.mus.max().round(3))
        print("Optimal action-values:", jnp.round(self.mus, 3))

        config = MultiArmedBanditConfig(k=k, initial_q_values=0, ucb_exploration=confidence_bound)
        if thompson_sampling:
            self.mab = MultiArmedBanditThompsonDynamic(config)
        elif confidence_bound is not None:
            self.mab = IncrementalMultiArmedBanditUCB(config)
        elif epsilon is not None:
            self.mab = IncrementalMultiArmedBanditEpsilonGreedy(config, epsilon=epsilon)
        else:
            self.mab = MultiArmedBandit(config)
        self.history = []
        self.reward_history = []

    def run(self, steps: int):
        for s in range(steps):
            action = self.mab.pick()
            reward = sample_gaussian(get_prng_key(self.prng_collection), self.mus[action], self.sigmas[action]).item()
            self.mab.update(action, reward)

            self.history.append(action)
            self.reward_history.append(reward)

            if s % 1000 == 999:
                print(f"Step {s + 1} completed")

        print("Learning complete")
        print(jnp.round(self.mab.estimates, 3))
        print(jnp.argmax(self.mab.estimates), jnp.max(self.mab.estimates))

    def plot_distribution(self):
        plt.figure()
        plt.scatter(range(len(self.history)), self.history)
        df = pd.DataFrame(zip(self.history, self.reward_history), columns=['action', 'reward'])
        df['average'] = df['reward'].cumsum() / (df.index+1)
        plt.figure()
        sb.violinplot(df, y='reward', x='action')
        plt.figure()
        plt.plot(df['average'])
        plt.plot(df.index, [self.mus.max()] * df.shape[0], 'r--')
        plt.show()


class GaussianBanditsComparator:
    prng_collection: str = "GaussianBanditsComparator"

    def __init__(self, k: int, epsilon: float = None, confidence_bound: float = None, mus: FloatArray = None, sigmas: FloatArray = None):
        self.mus = jnp.zeros((k,), dtype=jnp.float32) if mus is None else mus
        self.sigmas = jnp.ones((k,), dtype=jnp.float32) if sigmas is None else sigmas

        print("Best action and value:", self.mus.argmax(), self.mus.max().round(3))
        print("Optimal action-values:", jnp.round(self.mus, 3))

        config = MultiArmedBanditConfig(k=k, initial_q_values=0, ucb_exploration=confidence_bound)
        self.mabs = {
            "thompson": MultiArmedBanditThompson(config),
            "ucb": IncrementalMultiArmedBanditUCB(config),
            "epsilon": IncrementalMultiArmedBanditEpsilonGreedy(config, epsilon=epsilon),
            "greedy": MultiArmedBandit(config),
            "gradient": GradientMultiArmedBandit(config)
        }
        self.history = {name: [] for name in self.mabs}
        self.reward_history = {name: [] for name in self.mabs}

    def run(self, steps: int):
        for s in range(steps):
            rewards = sample_gaussian(get_prng_key(self.prng_collection), self.mus, self.sigmas)
            for name, mab in self.mabs.items():
                action = mab.pick()
                reward = rewards[action].item()
                mab.update(action, reward)
                self.history[name].append(action)
                self.reward_history[name].append(reward)

            if s % 1000 == 999:
                print(f"Step {s + 1} completed")

        print("Learning complete")
        for name in self.mabs:
            print(name)
            if name == 'gradient':
                print(jnp.round(self.mabs[name].preferences, 3))
                print(jnp.argmax(self.mabs[name].preferences), jnp.max(self.mabs[name].preferences))
            else:
                print(jnp.round(self.mabs[name].estimates, 3))
                print(jnp.argmax(self.mabs[name].estimates), jnp.max(self.mabs[name].estimates))

    def plot_distribution(self):
        plt.figure()
        for i, (name, history) in enumerate(self.history.items()):
            plt.scatter(range(len(history)), jnp.array(history)+(i*0.1), label=name)
        plt.legend()

        plt.figure()
        df = pd.DataFrame(
            zip(self.history.values(), self.reward_history.values()),
            columns=['action', 'reward'],
            index=self.mabs.keys())
        sb.violinplot(df.explode(['reward', 'action']), y='reward', x='action')

        plt.figure()
        for name in self.mabs:
            df = pd.DataFrame(zip(self.history[name], self.reward_history[name]), columns=['action', 'reward'])
            df['average'] = df['reward'].cumsum() / (df.index+1)
            plt.plot(df['average'], label=name)
        plt.plot(df.index, [self.mus.max()] * df.shape[0], 'r--')
        plt.legend()
        plt.show()


class ContextualGaussianBandits:
    prng_collection: str = 'ContextualGaussianBandits'

    def __init__(self,
                 k: int,
                 confidence_bound: float = None,
                 linucb_latent_dim: int = None,
                 linucb_hybrid: bool = False,
                 mus: FloatArray = None,
                 sigmas: FloatArray = None):
        self.mus = jnp.zeros((k,), dtype=jnp.float32) if mus is None else mus
        self.sigmas = jnp.ones((k,), dtype=jnp.float32) if sigmas is None else sigmas

        print("Best action and value:", self.mus.argmax(), self.mus.max().round(3))
        print("Optimal action-values:", jnp.round(self.mus, 3))

        self.features_dim = [k]
        self.history = []
        self.reward_history = []

        config = MultiArmedBanditConfig(k=k, ucb_exploration=confidence_bound)
        if confidence_bound is not None:
            if linucb_latent_dim is not None:
                if linucb_hybrid:
                    self.mab = LinUCBHybrid(config, linucb_latent_dim)
                    self.features_dim.append(k + linucb_latent_dim)
                else:
                    self.mab = LinUCB(config, linucb_latent_dim)
                    self.features_dim.append(linucb_latent_dim)

        if not hasattr(self, 'mab'):
            raise RuntimeError('No supported Contextual MAB')

    def run(self, steps: int):
        for s in range(steps):
            features = 1 + random.normal(get_prng_key(self.prng_collection), self.features_dim)
            action = self.mab.pick(features)
            reward = sample_gaussian(get_prng_key(self.prng_collection), self.mus[action], self.sigmas[action]).item()
            self.mab.update(action, reward, features=features)

            self.history.append(action)
            self.reward_history.append(reward)

            if s % 1000 == 999:
                print(f"Step {s + 1} completed")

        print("Learning complete")
        print(jnp.round(self.mab.estimates, 3))
        print(jnp.argmax(self.mab.estimates), jnp.max(self.mab.estimates))

    plot_distribution = GaussianBandits.plot_distribution


if __name__ == '__main__':
    k = 10
    params = {
        "confidence_bound": 1.25,
        "linucb_latent_dim": 20,
        "linucb_hybrid": True,
        "mus": random.normal(get_prng_key(), (k,), dtype=jnp.float32)
    }
    # testbed = GaussianBanditsComparator(k, **params)
    testbed = ContextualGaussianBandits(k, **params)
    testbed.run(5000)
    testbed.plot_distribution()

