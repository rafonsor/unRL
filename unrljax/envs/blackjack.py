import logging
from itertools import product

import gymnasium as gym

from unrljax import types as t
from unrljax.algos.monte_carlo import Action, State, OnPolicyFirstVisitMonteCarloControl, SAR, StateSet, ActionSet, \
    Trajectory

logger = logging.getLogger(__name__)


def make_blackjack(human: bool = False) -> t.Tuple[gym.Env, StateSet, ActionSet]:
    env = gym.make('Blackjack-v1', natural=False, render_mode="human" if human else None)
    stateset = frozenset([
        State(idx, str(state), False, state)
        for idx, state
        in enumerate(product(*(range(state.n) for state in env.observation_space)))
    ])
    actionset = frozenset([Action(idx, '', None) for idx in range(env.action_space.n)])
    return env, stateset, actionset


def run_episode(env: gym.Env, state_map: dict, action_map: dict) -> Trajectory:
    episode = []
    i = 1
    terminated = truncated = False

    observation, _ = env.reset()
    while not (terminated or truncated):
        state = state_map[observation]
        action = action_map[on.action(state.id)]

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
    env, stateset, actionset = make_blackjack(False)
    state_map = {s.representation: s for s in stateset}
    action_map = {a.id: a for a in actionset}

    on = OnPolicyFirstVisitMonteCarloControl(discount=0.9, epsilon=0.05, stateset=stateset, actionset=actionset)
    print(on.action_values)
    print(on.policy)

    episodes = []
    for ep in range(2500):
        logger.info(f'Starting episode {ep}')
        episode = run_episode(env, state_map, action_map)
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
            for s in stateset if s.representation[2] == third
        ))
        ax.set_title(f'Learnt state-values - {"" if third else "No "}Usable Ace')
        ax.plot_trisurf(x, y, z)
    plt.show()
