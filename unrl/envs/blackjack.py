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

import gymnasium as gym
import torch as pt

import unrl.types as t
from unrl.containers import Trajectory, Transition, FrozenTrajectory

logger = logging.getLogger(__name__)


def make_blackjack(human: bool = False) -> t.Tuple[gym.Env, int, int, t.Callable[[t.Tuple[int]], pt.Tensor]]:
    env = gym.make('Blackjack-v1', natural=False, render_mode="human" if human else None)
    num_state_dims = len(env.observation_space)  # noqa, observation_space is a tuple
    num_actions = env.action_space.n
    bounds = pt.Tensor([dim.n - 1 for dim in env.observation_space])

    def transform(obs: t.Tuple[int]) -> pt.Tensor:
        return pt.Tensor(obs) / bounds

    return env, num_state_dims, num_actions, transform


def run_episode(env: gym.Env, transform: t.Callable[[t.Tuple[int]], pt.Tensor], random_play: bool = False) -> t.Tuple[Trajectory, float]:
    observation, _ = env.reset()
    state = transform(observation)
    terminated = truncated = False
    episode = []
    i = 0

    while not (terminated | truncated):
        if random_play:
            action = env.action_space.sample()
        else:
            action = model.sample(state)
        logger.debug(f'Step {i}: Applying action {action} in state {state}')
        observation, reward, terminated, truncated, _ = env.step(action)
        next_state = transform(observation)
        episode.append(Transition(state, action, reward, next_state))
        if env.render_mode == 'human':
            from time import sleep
            sleep(0.25)

    loss = model.optimise(FrozenTrajectory.from_trajectory(episode))

    if terminated:
        final_reward = episode[-1].reward
        if final_reward == 1:
            logger.info(f'Won the game after {len(episode)} steps.')
        elif final_reward == 0:
            logger.info(f'Reached a draw after {len(episode)} steps.')
        else:
            logger.info(f'Lost the game after {len(episode)} steps.')
    elif truncated:
        logger.info('Episode did not complete within the allowed steps.')

    logger.debug(f"Episode average loss = {loss:.6f}")
    return episode, loss


if __name__ == '__main__':
    import datetime as dt
    from itertools import product

    from matplotlib import pyplot as plt

    from unrl.envs.test_models import prepare_game_model_reinforce

    logging.basicConfig(level=logging.INFO)
    logger.setLevel(logging.DEBUG)

    env, num_state_dims, num_actions, obs_to_state = make_blackjack(human=False)
    model = prepare_game_model_reinforce(num_state_dims, num_actions, baseline=True)

    num_episodes = 250
    logger.info(f'Playing Blackjack for {num_episodes} episodes')

    rewards = []
    losses = []
    for ep in range(num_episodes):
        logger.debug("starting new episode")
        (episode, loss) = run_episode(env, obs_to_state, not len(losses))
        rewards.append(episode[-1].reward)
        losses.append(loss)
        if (ep+1) % 50 == 0:
            logger.info(f"[{dt.datetime.now().isoformat()}Episode {ep+1} done")

    # Plot accumulated rewards
    plt.plot(pt.cumsum(pt.Tensor(rewards), 0))
    plt.ylabel('Accumulated reward')
    plt.xlabel('Episode #')
    plt.ylim([-num_episodes, num_episodes])
    axy = plt.twinx()
    axy.plot(pt.Tensor(losses).cumsum(0)/pt.arange(1, num_episodes+1), c='r')
    axy.scatter(range(num_episodes), losses, c='r', s=3)
    axy.set_ylabel('Average episode loss')
    plt.tight_layout()

    if not hasattr(model, 'state_value_model'):
        plt.show()
        exit(0)

    def plot_learnt_state_values(estimates: pt.Tensor, ace: bool):
        variant = f'{"" if ace else "No "}Usable Ace'
        logger.info("Plotting Learnt State-values")
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.plot_trisurf(estimates[:, 0], estimates[:, 1], estimates[:, 2])
        ax.set_title(f'Learnt state-values - {variant}')
        ax.set_xlabel("Player's Current Sum")
        ax.set_ylabel("Dealer's First Card")
        ax.set_zlabel("Value Estimate")
        ax.set_zlim([-1, 1])

    # Retrieve state-value estimates for entire state space
    stateset = pt.Tensor([
        [current_sum_player, first_card_dealer, 0]
        for current_sum_player, first_card_dealer
        in product(range(env.observation_space[0].n), range(env.observation_space[1].n))
    ])
    estimates_ace = stateset.clone()
    estimates_ace[:, 2:] = model.state_value_model(stateset).detach()
    stateset[:, 2] = 1
    estimates_no_ace = stateset.clone()
    estimates_no_ace[:, 2:] = model.state_value_model(stateset).detach()

    # Plot learnt state-values
    plot_learnt_state_values(estimates_ace, True)
    plot_learnt_state_values(estimates_no_ace, False)
    plt.show()
