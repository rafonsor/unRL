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
from unrl.containers import Trajectory

logger = logging.getLogger(__name__)


def make_cliff_walking(human: bool = False) -> t.Tuple[gym.Env, int, int, t.Callable[[int], pt.Tensor]]:
    env = gym.make('CliffWalking-v0', render_mode="human" if human else None)
    num_state_dims = env.observation_space.n  # noqa, observation_space is a tuple
    num_actions = env.action_space.n

    def transform(obs: int) -> pt.Tensor:
        return pt.nn.functional.one_hot(pt.Tensor((obs,)).long(), num_state_dims)[0].float()

    return env, num_state_dims, num_actions, transform


def run_episode(model, env: gym.Env, transform: t.Callable[[int], pt.Tensor], max_episode_length: int) -> t.Tuple[Trajectory, float]:
    observation, _ = env.reset()
    state = transform(observation)
    terminated = truncated = False
    info = None
    i = 1

    gen = model.online_optimise(state)
    while (action := gen.send(info)) is not None and (i := i+1):
        logger.debug(f'Step {i}: Applying action {action.item()} in state {observation}')
        observation, reward, terminated, truncated, _ = env.step(action.item())
        state = transform(observation)
        if i > max_episode_length:
            truncated = True
        info = (reward, state, terminated, terminated | truncated)

    (episode, loss) = gen.value

    if terminated:
        logger.info(f'Reached destination {len(episode)} steps.')
    elif truncated:
        logger.info('Episode did not complete within the allowed steps.')

    logger.debug(f"Episode average loss = {loss:.6f}")
    return episode, loss


if __name__ == '__main__':
    import datetime as dt

    from matplotlib import pyplot as plt

    from unrl.envs.test_models import prepare_game_model_actorcritic

    logging.basicConfig(level=logging.INFO)
    logger.setLevel(logging.DEBUG)

    env, num_state_dims, num_actions, obs_to_state = make_cliff_walking(human=False)
    model = prepare_game_model_actorcritic(num_state_dims, num_actions, advantage=True, eligibility_traces=False)

    num_episodes = 100
    logger.info(f'Playing CliffWalking for {num_episodes} episodes')

    rewards = []
    losses = []
    for ep in range(num_episodes):
        logger.debug("starting new episode")
        (episode, loss) = run_episode(model, env, obs_to_state, max_episode_length=1000)
        rewards.append(episode[-1].reward)
        losses.append(loss)
        if (ep+1) % 10 == 0:
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
    plt.show()
