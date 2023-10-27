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
from unrl.action_sampling import make_sampler, ActionSamplingMode
from unrl.algos.policy_gradient import ActorCritic, Policy, EligibilityTraceActorCritic
from unrl.containers import Trajectory

logger = logging.getLogger(__name__)


def make_blackjack(human: bool = False) -> t.Tuple[gym.Env, t.Callable[[t.Tuple[int]], pt.Tensor]]:
    env = gym.make('Blackjack-v1', natural=False, render_mode="human" if human else None)
    bounds = pt.Tensor([dim.n - 1 for dim in env.observation_space])

    def transform(obs: t.Tuple[int]) -> pt.Tensor:
        return pt.Tensor(obs) / bounds

    return env, transform


def run_episode(env: gym.Env, transform: t.Callable[[t.Tuple[int]], pt.Tensor]) -> Trajectory:
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
        x = F.relu(self.layer1(state))
        x = F.relu(self.layer2(x))
        return self.logsm(self.layer3(x))


class ExampleStateValueModel(pt.nn.Module):
    def __init__(self, num_state_dims, hidden_dim: int):
        super().__init__()
        self.layer1 = pt.nn.Linear(num_state_dims, hidden_dim)
        self.layer2 = pt.nn.Linear(hidden_dim, hidden_dim // 3)
        self.layer3 = pt.nn.Linear(hidden_dim // 3, 1)

    def forward(self, state: pt.Tensor) -> pt.Tensor:
        x = F.relu(self.layer1(state))
        x = F.relu(self.layer2(x))
        estimate = F.tanh(self.layer3(x))
        return estimate


def prepare_game_model(env: gym.Env, eligibility_traces: bool = False) -> ActorCritic:
    discount_factor = 0.99
    learning_rate_policy = 1e-6
    learning_rate_values = 1e-6
    trace_decay_policy = 0.1
    trace_decay_values = 0.1
    weight_decay_policy = 0.1
    weight_decay_values = 0.1
    hidden_dim_policy = 10
    hidden_dim_values = 30
    num_state_dims = len(env.observation_space)  # noqa, observation_space is a tuple
    num_actions = env.action_space.n
    policy = ExamplePolicy(num_state_dims, num_actions, hidden_dim_policy)
    state_value_model = ExampleStateValueModel(num_state_dims, hidden_dim_values)
    action_sampler = make_sampler(ActionSamplingMode.EPSILON_GREEDY)
    if eligibility_traces:
        actor_critic = EligibilityTraceActorCritic(
            policy, state_value_model,
            discount_factor=discount_factor,
            learning_rate_policy=learning_rate_policy,
            learning_rate_values=learning_rate_values,
            trace_decay_policy=trace_decay_policy,
            trace_decay_values=trace_decay_values,
            weight_decay_policy=weight_decay_policy,
            weight_decay_values=weight_decay_values,
            action_sampler=action_sampler)
    else:
        actor_critic = ActorCritic(
            policy, state_value_model, discount_factor=discount_factor, learning_rate_policy=learning_rate_policy,
            learning_rate_values=learning_rate_values, action_sampler=action_sampler)
    return actor_critic


if __name__ == '__main__':
    import datetime as dt
    from itertools import product

    from matplotlib import pyplot as plt

    logging.basicConfig(level=logging.INFO)
    logger.setLevel(logging.DEBUG)

    env, obs_to_state = make_blackjack(human=False)
    model = prepare_game_model(env, eligibility_traces=True)

    num_episodes = 50000
    logger.info(f'Playing Blackjack for {num_episodes} episodes')

    rewards = []
    for ep in range(num_episodes):
        logger.debug("starting new episode")
        rewards.append(run_episode(env, obs_to_state)[-1].reward)
        if (ep+1) % 5000 == 0:
            logger.info(f"[{dt.datetime.now().isoformat()}Episode {ep+1} done")

    # Plot accumulated rewards
    plt.plot(pt.cumsum(pt.Tensor(rewards), 0))
    plt.ylabel('Accumulated reward')
    plt.xlabel('Episode #')
    plt.ylim([-num_episodes, num_episodes])

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
