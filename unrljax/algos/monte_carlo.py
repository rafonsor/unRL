from dataclasses import dataclass

from jax import numpy as jnp

import unrljax.types as t
from unrljax.basic import argmax_random
from unrljax.random import random, get_prng_key

Reward = float


@dataclass(unsafe_hash=True)
class Action:
    id: int
    name: str
    representation: t.Any = None


@dataclass(unsafe_hash=True)
class State:
    id: int
    name: str
    terminal: bool
    representation: t.Any = None


@dataclass
class SAR:
    state: State
    action: Action
    reward: Reward

    def __iter__(self) -> t.List[State | Action | Reward]:
        yield self.state
        yield self.action
        yield self.reward


@dataclass
class SARS(SAR):
    successor: State

    def __iter__(self) -> t.List[State | Action | Reward]:
        yield from super().__iter__()
        yield self.successor


ActionSet = t.FrozenSet[Action]
StateSet = t.FrozenSet[State]
Trajectory = t.Sequence[t.Tuple[State, Action, Reward] | SAR | SARS]


class OnPolicyFirstVisitMonteCarloControl:

    action_values: t.FloatArray

    def __init__(self, discount: float, epsilon: float, stateset: StateSet, actionset: ActionSet):
        self.discount = discount
        self.epsilon = epsilon
        self.marginal = epsilon / len(actionset)
        self.policy = jnp.ones((len(stateset), len(actionset))) / len(actionset)
        self.action_values = random.normal(get_prng_key(), (len(stateset), len(actionset)))
        self.returns = jnp.zeros((len(stateset), len(actionset)))
        self.counts = jnp.zeros((len(stateset), len(actionset)))

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


if __name__ == '__main__':
    stateset = frozenset([State(idx, '', False, None) for idx in range(10)] + [State(10, '', True, None)])
    actionset = frozenset([Action(idx, '', None) for idx in range(4)])
    on = OnPolicyFirstVisitMonteCarloControl(discount=0.9, epsilon=0.05, stateset=stateset, actionset=actionset)

    episode = [
        SAR(State(0, '', False, None), Action(0, '', None), -1),
        SAR(State(4, '', False, None), Action(2, '', None), -1),
        SAR(State(7, '', False, None), Action(2, '', None), -1),
        SAR(State(9, '', False, None), Action(1, '', None), -1),
        SAR(State(1, '', False, None), Action(3, '', None), 0),
    ]
    print(on.action_values)
    print(on.policy)
    on.batch_optimise([episode] * 10)
    print(on.action_values)
    print(on.policy)
