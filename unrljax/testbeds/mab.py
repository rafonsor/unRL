
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
from jax import numpy as jnp
from jax import random

from unrljax.algos.mab import MultiArmedBanditConfig, IncrementalMultiArmedBanditEpsilonGreedy, \
    IncrementalMultiArmedBanditUCB, MultiArmedBandit, MultiArmedBanditThompson, MultiArmedBanditThompsonDynamic, \
    GradientMultiArmedBandit
from unrljax.basic import sample_gaussian
from unrljax.random import get_prng_key
from unrljax.types import FloatArray


class GaussianBandits:
    prng_collection: str = 'GaussianBandits'

    def __init__(self, k: int, epsilon: float = None, confidence_bound: float = None, thompson_sampling: bool = False, mus: FloatArray = None, sigmas: FloatArray = None):
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

            if s % 500 == 499:
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


if __name__ == '__main__':
    k = 10
    testbed = GaussianBanditsComparator(k, epsilon=0.05, confidence_bound=1.25, mus=random.normal(get_prng_key(), (k,), dtype=jnp.float32))
    testbed.run(5000)
    testbed.plot_distribution()

