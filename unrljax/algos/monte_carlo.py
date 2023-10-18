from jax import numpy as jnp

import unrljax.types as t
from unrljax.basic import argmax_random, argmax_all
from unrljax.objects import DiscreteStateSet, DiscreteActionSet, Trajectory
from unrljax.random import random, get_prng_key


class OnPolicyFirstVisitMonteCarloControl:
    """On-policy Monte Carlo Control following first-visit updates for discrete State and Action spaces

    Reference:
    [1] Sutton, R. S., & Barto, A. G. (2018). Section 5.4., Reinforcement learning: An introduction (2nd ed.). The MIT
        Press.
    """
    action_values: t.FloatArray

    def __init__(self, discount: float, epsilon: float, stateset: DiscreteStateSet, actionset: DiscreteActionSet):
        self.discount = discount
        self.epsilon = epsilon
        self.num_actions = len(actionset)
        self.marginal = epsilon / self.num_actions
        self.policy = jnp.ones((len(stateset), self.num_actions)) / self.num_actions
        self.action_values = random.normal(get_prng_key(), (len(stateset), self.num_actions))
        self.returns = jnp.zeros((len(stateset), self.num_actions))
        self.counts = jnp.zeros((len(stateset), self.num_actions))

    def action(self, state: int) -> int:
        if random.normal(get_prng_key()) > self.epsilon:
            return argmax_random(get_prng_key(), self.policy[state])
        return random.randint(get_prng_key(), (1,), 0, self.num_actions - 1).item()

    def optimise(self, episode: Trajectory):
        G = 0

        T = len(episode) - 1
        first_visits = set()
        first_visit_timesteps = set()
        for timestep, (state, action, _) in enumerate(episode):
            sa = (state, action)
            if sa in first_visits:
                continue
            first_visits.add(sa)
            first_visit_timesteps.add(timestep)
        del first_visits

        for offset, (state, action, reward) in enumerate(episode[::-1]):
            G = self.discount * G + reward
            if (T-offset) not in first_visit_timesteps:
                continue

            # Update Q-values
            self.counts = self.counts.at[state.id, action.id].add(1)
            self.returns = self.returns.at[state.id, action.id].add(G)
            self.action_values = self.action_values.at[state.id, action.id].set(
                self.returns[state.id, action.id] / self.counts[state.id, action.id])

            # Update policy
            astar = argmax_random(get_prng_key(), self.action_values[state.id])
            self.policy = self.policy.at[state.id].set(self.marginal)
            self.policy = self.policy.at[state.id, astar].add(1 - self.epsilon)

    def batch_optimise(self, episodes: t.Sequence[Trajectory]):
        for ep in episodes:
            self.optimise(ep)


class OffPolicyMonteCarloControl:
    """Off-policy Monte Carlo Control for discrete State and Action spaces

    Reference:
    [1] Sutton, R. S., & Barto, A. G. (2018). Section 5.7., Reinforcement learning: An introduction (2nd ed.). The MIT
        Press.
    """
    action_values: t.FloatArray
    target_policy: t.FloatArray
    counts: t.IntArray

    def __init__(self, discount: float, stateset: DiscreteStateSet, actionset: DiscreteActionSet):
        self.discount = discount
        self.num_actions = len(actionset)
        self.action_values = random.normal(get_prng_key(), (len(stateset), self.num_actions))
        policy = jnp.where(self.action_values == self.action_values.max(axis=1), 1, 0)
        self.target_policy = policy / policy.sum(axis=1)[:, None]

        self.counts = jnp.zeros((len(stateset), self.num_actions))

    def get_policy(self, randomised: bool = False) -> t.FloatArray:
        if randomised:
            policy = random.uniform(get_prng_key(), self.target_policy.shape)
            return policy / policy.sum(axis=1)
        return self.target_policy.copy()

    def action(self, state: int, policy: t.Optional[t.FloatArray] = None) -> int:
        policy = policy or self.target_policy
        return argmax_random(get_prng_key(), policy[state])

    def optimise(self, episode: Trajectory, behaviour_policy: t.FloatArray):
        assert behaviour_policy.shape == self.target_policy.shape
        G = 0
        W = 1

        for offset, (state, action, reward) in enumerate(episode[::-1]):
            self.counts = self.counts.at[state.id, action.id].add(W)

            # Update Q-values
            G = self.discount * G + reward
            delta = W * (G - self.action_values[state.id, action.id]) / self.counts[state.id, action.id]
            self.action_values = self.action_values.at[state.id, action.id].set(delta)

            # Update policy
            self.target_policy = self.target_policy.at[state.id].set(0)
            astars = argmax_all(self.action_values[state.id])
            if action.id not in astars:
                self.target_policy = self.target_policy.at[state.id, random.choice(get_prng_key(), astars)].set(1)
                break
            self.target_policy = self.target_policy.at[state.id, action.id].set(1)
            W /= behaviour_policy[state.id, action.id]

    def batch_optimise(self, episodes: t.Sequence[Trajectory], behaviour_policy: t.FloatArray):
        for ep in episodes:
            self.optimise(ep, behaviour_policy)


if __name__ == '__main__':
    from unrljax.objects import State, Action, SAR
    stateset_ = DiscreteStateSet([State(idx, '', False, None) for idx in range(10)] + [State(10, '', True, None)])
    actionset_ = DiscreteActionSet([Action(idx, '', None) for idx in range(4)])
    on = OnPolicyFirstVisitMonteCarloControl(discount=0.9, epsilon=0.05, stateset=stateset_, actionset=actionset_)

    episode_ = [
        SAR(State(0, '', False, None), Action(0, '', None), -1),
        SAR(State(4, '', False, None), Action(2, '', None), -1),
        SAR(State(7, '', False, None), Action(2, '', None), -1),
        SAR(State(9, '', False, None), Action(1, '', None), -1),
        SAR(State(1, '', False, None), Action(3, '', None), 0),
    ]
    print(on.action_values)
    print(on.policy)
    on.batch_optimise([episode_] * 10)
    print(on.action_values)
    print(on.policy)
