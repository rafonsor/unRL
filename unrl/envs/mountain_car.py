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
import torch.nn.functional as F

import unrl.types as t
from unrl.algos.policy_gradient import ActorCritic, Policy, EligibilityTraceActorCritic
from unrl.containers import Trajectory

logger = logging.getLogger(__name__)


def make_mountain_car(human: bool = False) -> t.Tuple[gym.Env, t.Callable[[t.NDArray], pt.Tensor]]:
    env = gym.make('MountainCar-v0', render_mode="human" if human else None)
    bounds = pt.Tensor(env.observation_space.high - env.observation_space.low)

    def transform(obs: t.NDArray) -> pt.Tensor:
        return (pt.Tensor(obs) + env.observation_space.high) / bounds

    return env, transform


def run_episode(env: gym.Env, transform: t.Callable[[t.NDArray], pt.Tensor]) -> Trajectory:
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
        info = (reward, state, terminated | truncated)

    episode = gen.value

    if terminated:
        logger.info(f'Won the game after {len(episode)} steps.')
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
        x = F.relu(self.layer1(state))
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
        estimate = -F.relu(self.layer3(x))
        return estimate


def prepare_game_model(env: gym.Env, eligibility_traces: bool = False) -> ActorCritic:
    discount_factor = 0.9
    learning_rate_policy = 1e-8
    learning_rate_values = 1e-8
    trace_decay_policy = 0.10
    trace_decay_values = 0.10
    hidden_dim_policy = 10
    hidden_dim_values = 10
    num_state_dims = env.observation_space.shape[0]  # noqa, observation_space is a tuple
    num_actions = env.action_space.n
    policy = ExamplePolicy(num_state_dims, num_actions, hidden_dim_policy)
    state_value_model = ExampleStateValueModel(num_state_dims, hidden_dim_values)
    if eligibility_traces:
        actor_critic = EligibilityTraceActorCritic(
            policy, state_value_model, discount_factor, learning_rate_policy, learning_rate_values, trace_decay_policy,
            trace_decay_values)
    else:
        actor_critic = ActorCritic(
            policy, state_value_model, learning_rate_policy, learning_rate_values, discount_factor)
    return actor_critic


if __name__ == '__main__':
    import datetime as dt

    from matplotlib import pyplot as plt

    logging.basicConfig(level=logging.INFO)
    logger.setLevel(logging.DEBUG)

    env, obs_to_state = make_mountain_car(human=False)
    model = prepare_game_model(env, eligibility_traces=True)

    num_episodes = 200
    logger.info(f'Playing MountainCar for {num_episodes} episodes')

    rewards = []
    for ep in range(num_episodes):
        logger.debug("starting new episode")
        rewards.append(-len(run_episode(env, obs_to_state)))
        if (ep+1) % 1000 == 0:
            logger.info(f"[{dt.datetime.now().isoformat()}Episode {ep+1} done")

    # Plot accumulated rewards
    plt.plot([0, num_episodes-1], [pt.Tensor(rewards).mean().item()]*2)
    plt.scatter(range(num_episodes), rewards)
    plt.ylabel('Episode penalty')
    plt.xlabel('Episode #')
    plt.ylim([-201, 1])
    plt.show()
