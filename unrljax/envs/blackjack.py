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
import logging
from itertools import product

import gymnasium as gym

import unrljax.types as t
from unrljax.algos.monte_carlo import Action, State, OnPolicyFirstVisitMonteCarloControl, SAR, DiscreteStateSet, \
    DiscreteActionSet, Trajectory

logger = logging.getLogger(__name__)


def make_blackjack(human: bool = False) -> t.Tuple[gym.Env, DiscreteStateSet, DiscreteActionSet]:
    env = gym.make('Blackjack-v1', natural=False, render_mode="human" if human else None)
    stateset = DiscreteStateSet([
        State(idx, str(state), False, state)
        for idx, state
        in enumerate(product(*(range(state.n) for state in env.observation_space)))
    ])
    actionset = DiscreteActionSet([Action(idx, '', None) for idx in range(env.action_space.n)])
    return env, stateset, actionset


def run_episode(env: gym.Env, statespace: DiscreteStateSet, actionspace: DiscreteActionSet) -> Trajectory:
    episode = []
    i = 1
    terminated = truncated = False

    observation, _ = env.reset()
    while not (terminated or truncated):
        state = statespace.by_representation(observation)
        action = actionspace.by_id(on.action(state.id))

        logger.debug(f'Step {i}: Applying action {action.id} in state {state.id}')
        observation, reward, terminated, truncated, _ = env.step(action.id)
        episode.append(SAR(state, action, reward))
        i += 1

    if terminated:
        logger.debug('Reached a terminal state')
    elif truncated:
        logger.debug('Episode has concluded')
    return episode


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    env, stateset_, actionset_ = make_blackjack(False)

    on = OnPolicyFirstVisitMonteCarloControl(discount=0.9, epsilon=0.05, stateset=stateset_, actionset=actionset_)
    print(on.action_values)
    print(on.policy)

    episodes = []
    for ep in range(25):
        logger.info(f'Starting episode {ep}')
        episode = run_episode(env, stateset_, actionset_)
        episodes.append(episode)
        on.optimise(episode)

    logger.info(f'Replaying batch of episodes')
    on.batch_optimise(episodes)
    print(on.action_values)
    print(on.policy)

    logger.info("Plotting state-values derived from learnt action-values")
    import matplotlib.pyplot as plt
    for third in range(env.observation_space[2].n):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        x, y, z = zip(*(
            (s.representation[0], s.representation[1], (on.action_values[s.id] * on.policy[s.id]).mean())
            for s in stateset_ if s.representation[2] == third
        ))
        ax.set_title(f'Learnt state-values - {"" if third else "No "}Usable Ace')
        ax.plot_trisurf(x, y, z)
    plt.show()
