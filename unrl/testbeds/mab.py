import random
import typing as t

import matplotlib.pyplot as plt
import seaborn as sb

from unrl.algos.mab import MultiArmedBanditConfig, IncrementalMultiArmedBanditEpsilonGreedy, \
    IncrementalMultiArmedBanditUCB
from unrl.basic import sample_gaussian, argmax


class GaussianBandits:
    def __init__(self, k: int, epsilon: float = None, mus: t.List[float] = None, sigmas: t.List[float] = None):
        self.mus = mus or [0.0] * k
        self.sigmas = sigmas or [1.0] * k

        print("Optimal action-values:", [round(q, 3) for q in self.mus])
        print(argmax(self.mus), max(self.mus))

        config = MultiArmedBanditConfig(k=k, initial_q_values=0, ucb_exploration=2)
        if epsilon is None:
            # self.mab = IncrementalMultiArmedBandit(config)
            self.mab = IncrementalMultiArmedBanditUCB(config)
        else:
            self.mab = IncrementalMultiArmedBanditEpsilonGreedy(config, epsilon=epsilon)
        self.history = []
        self.reward_history = []

    def run(self, steps: int):
        for s in range(steps):
            action = self.mab.pick()
            reward = sample_gaussian(self.mus[action], self.sigmas[action])
            self.mab.update(action, reward)

            self.history.append(action)
            # self.reward_history[action].append(reward)
            self.reward_history.append(reward)

            if s % 1000 == 999:
                print(f"Step {s + 1} completed")

        print("Learning complete")
        print([round(q, 3) for q in self.mab.estimates])
        print(argmax(self.mab.estimates), max(self.mab.estimates))

    def plot_distribution(self):
        plt.figure()
        plt.scatter(range(len(self.history)), self.history)
        import pandas as pd
        df = pd.DataFrame(zip(self.history, self.reward_history), columns=['action', 'reward'])
        df['average'] = df['reward'].cumsum() / (df.index+1)
        plt.figure()
        sb.violinplot(df, y='reward', x='action')
        plt.figure()
        plt.plot(df['average'])
        plt.show()


if __name__ == '__main__':
    k = 20
    testbed = GaussianBandits(k, epsilon=None, mus=[random.normalvariate(0.0, 1) for _ in range(k)])
    testbed.run(10000)
    testbed.plot_distribution()

