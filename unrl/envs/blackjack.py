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
import torch as pt
import torch.nn.functional as F

import unrl.types as t
from unrl.algos.policy_gradient import ActorCritic, Policy
from unrl.containers import Trajectory

logger = logging.getLogger(__name__)


def make_blackjack(human: bool = False) -> gym.Env:
    return gym.make('Blackjack-v1', natural=False, render_mode="human" if human else None)


def run_episode(env: gym.Env) -> Trajectory:
    observation, _ = env.reset()
    state = pt.Tensor(observation)
    terminated = truncated = False
    info = None
    i = 0

    gen = model.online_optimise(state)
    while (action := gen.send(info)) is not None and (i := i+1):
        logger.debug(f'Step {i}: Applying action {action.item()} in state {state}')
        observation, reward, terminated, truncated, _ = env.step(action.item())
        state = pt.Tensor(observation)
        info = (reward, state, terminated | truncated)
        if env.render_mode == 'human':
            from time import sleep
            sleep(0.5)

    episode = gen.value

    if terminated:
        final_reward = info[0]
        if final_reward == 1:
            logger.info(f'Won the game after {len(episode)} steps.')
        elif final_reward == 0:
            logger.info(f'Reached a draw after {len(episode)} steps.')
        else:
            logger.info(f'Lost the game after {len(episode)} steps.')
    elif truncated:
        logger.info('Episode did not complete within the allowed steps.')

    return episode


class ExamplePolicy(Policy):
    def __init__(self, num_state_dims: int, num_actions: int, hidden_dim: int):
        super().__init__()
        self.layer1 = pt.nn.Linear(num_state_dims, hidden_dim)
        self.layer2 = pt.nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = pt.nn.Linear(hidden_dim, num_actions)
        self.logsm = pt.nn.LogSoftmax(0)

    def forward(self, state: pt.Tensor) -> pt.Tensor:
        x = F.sigmoid(self.layer1(state))
        x = F.relu(self.layer2(x))
        return self.logsm(self.layer3(x))


class ExampleStateValueModel(pt.nn.Module):
    def __init__(self, num_state_dims, hidden_dim: int):
        super().__init__()
        self.layer1 = pt.nn.Linear(num_state_dims, hidden_dim)
        self.layer2 = pt.nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = pt.nn.Linear(hidden_dim, 1)

    def forward(self, state: pt.Tensor) -> pt.Tensor:
        x = F.relu(self.layer1(state))
        x = F.relu(self.layer2(x))
        estimate = self.layer3(x)
        return estimate


def prepare_game_model(env: gym.Env) -> ActorCritic:
    discount_factor = 0.9
    learning_rate_policy = 1e-4
    learning_rate_values = 1e-4
    hidden_dim_policy = 100
    hidden_dim_values = 100
    num_state_dims = len(env.observation_space)  # noqa, observation_space is a tuple
    num_actions = env.action_space.n
    policy = ExamplePolicy(num_state_dims, num_actions, hidden_dim_policy)
    state_value_model = ExampleStateValueModel(num_state_dims, hidden_dim_values)
    actor_critic = ActorCritic(policy, state_value_model, learning_rate_policy, learning_rate_values, discount_factor)
    return actor_critic


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger.setLevel(logging.DEBUG)

    env = make_blackjack(human=True)
    model = prepare_game_model(env)

    num_episodes = 20
    logger.info(f'Playing Blackjack for {num_episodes} episodes')

    for _ in range(num_episodes):
        logger.debug("starting new episode")
        run_episode(env)
