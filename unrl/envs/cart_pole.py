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
from unrl.containers import SARSTrajectory

logger = logging.getLogger(__name__)


def make_cart_pole(human: bool = False) -> t.Tuple[gym.Env, int, int, t.Callable[[t.NDArray], pt.Tensor]]:
    env = gym.make('CartPole-v1', render_mode="human" if human else None)
    num_state_dims = env.observation_space.shape[0]
    num_actions = env.action_space.n

    def transform(obs: t.NDArray) -> pt.Tensor:
        return pt.Tensor(obs)

    return env, num_state_dims, num_actions, transform


def run_episode(env: gym.Env, transform: t.Callable[[t.NDArray], pt.Tensor]) -> t.Tuple[SARSTrajectory, float]:
    observation, _ = env.reset()
    state = transform(observation)
    terminated = truncated = False
    info = None
    i = 0

    gen = model.online_optimise(state)
    while (action := gen.send(info)) is not None and (i := i+1):
        logger.debug(f'Step {i}: Applying action {action.item()} in state {state}')
        observation, reward, terminated, truncated, _ = env.step(action.item())
        state = transform(observation)
        info = (reward, state, terminated, terminated | truncated)
        if env.render_mode == 'human':
            from time import sleep
            sleep(0.01)

    (episode, loss) = gen.value

    if terminated:
        logger.info(f'Concluded the game after {len(episode)} steps.')
    elif truncated:
        logger.info('Episode did not complete within the allowed steps.')

    logger.debug(f"Episode average loss = {loss:.6f}")
    return episode, loss


if __name__ == '__main__':
    import datetime as dt

    from matplotlib import pyplot as plt

    from unrl.envs.test_models import prepare_game_model_dqn

    logging.basicConfig(level=logging.INFO)
    logger.setLevel(logging.DEBUG)

    env, num_state_dims, num_actions, obs_to_state = make_cart_pole(human=False)
    model = prepare_game_model_dqn(num_state_dims, num_actions, buffer_size=10000, priority=True, double=True)

    num_episodes = 10000
    logger.info(f'Playing CartPole for {num_episodes} episodes')

    rewards = []
    losses = []
    for ep in range(num_episodes):
        logger.debug("starting new episode")
        (episode, loss) = run_episode(env, obs_to_state)
        rewards.append(len(episode))
        del episode
        losses.append(loss)
        if (ep+1) % 1000 == 0:
            logger.info(f"[{dt.datetime.now().isoformat()}Episode {ep+1} done")

    # Plot accumulated rewards
    plt.plot([0, num_episodes-1], [pt.Tensor(rewards).mean().item()]*2)
    plt.scatter(range(num_episodes), rewards)
    plt.ylabel('Episode reward')
    plt.xlabel('Episode #')
    plt.ylim([0, 500])
    axy = plt.twinx()
    axy.plot(pt.Tensor(losses).cumsum(0)/pt.arange(1, num_episodes+1), c='r')
    axy.scatter(range(num_episodes), losses, c='r', s=3)
    axy.set_ylabel('Average episode loss')
    plt.tight_layout()
    plt.show()
