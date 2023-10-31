#  Copyright 2023 The unRL Authors.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
from dataclasses import dataclass

import torch as pt

import unrl.types as t


@dataclass
class Transition:
    state: pt.Tensor
    action: t.IntLike
    reward: t.FloatLike
    next_state: pt.Tensor

    def __iter__(self) -> t.Generator[t.Any, None, None]:
        return (getattr(self, field) for field in self.__class__.__dataclass_fields__)


@dataclass
class ContextualTransition(Transition):
    terminates: bool


Trajectory: t.TypeAlias = t.Sequence[Transition]
ContextualTrajectory: t.TypeAlias = t.Sequence[ContextualTransition]


# TODO: Improvement (2023-10-24)
# Change the implementation of FrozenTrajectory to support operating as a linked list where `next_state` is an alias to
# the next transition's `state`. Note, albeit such is a more compressed representation, doing so effectively couples
# all transitions, some uses of FrozenTrajectory containers may prefer independence between transitions.
class FrozenTrajectory:
    n: int
    __states: pt.Tensor
    __actions: pt.Tensor
    __rewards: pt.Tensor
    __next_states: t.Optional[pt.Tensor]
    __index: t.List[int]

    def __init__(self, states: pt.Tensor, actions: pt.Tensor, rewards: pt.Tensor, next_states: t.Optional[pt.Tensor]):
        self.__n = states.shape[0]
        assert self.__n == actions.shape[0] and self.__n == rewards.shape[0], "Length of inputs must match"
        if next_states is not None:
            # Ensure there are next states for all transitions until at least the penultimate. The last transition is
            # allowed to not contain a `next_state` since it may refer to a goal state without any successor state.
            assert (next_states.shape[0] - 1) <= self.__n <= next_states.shape[0], \
                "Length of next_states must match other inputs, or at least be specified for all transition but the " \
                "last."
        self.__states = states
        self.__actions = actions
        self.__rewards = rewards
        self.__next_states = next_states
        self.__index = list(range(self.__n))

    @classmethod
    def from_trajectory(cls, trajectory: Trajectory) -> "FrozenTrajectory":
        n = len(trajectory)
        assert n, "Cannot instantiate a FrozenTrajectory from an empty Trajectory"
        states, actions, rewards, next_states = [], [], [], []
        for transition in trajectory:
            states.append(transition.state)
            actions.append(transition.action)
            rewards.append(transition.reward)
            next_states.append(transition.next_state)

        states = pt.stack(states)
        actions = pt.concat(actions) if pt.is_tensor(trajectory[0].action) else pt.Tensor(actions)
        actions = actions.type(pt.int)
        rewards = pt.concat(rewards) if pt.is_tensor(trajectory[0].reward) else pt.Tensor(rewards)
        if pt.is_tensor(trajectory[0].next_state):
            if pt.is_tensor(trajectory[-1].next_state):
                next_states = pt.stack(next_states)
            else:
                next_states = pt.stack(next_states[:-1])

        return cls(states, actions, rewards, next_states)

    @classmethod
    def from_tuples(cls, trajectory: t.Sequence[t.SoftSARS]) -> "FrozenTrajectory":
        n = len(trajectory)
        assert n, "Cannot instantiate a FrozenTrajectory from an empty list of transitions"
        _, action, reward, next_state = trajectory[0]
        states, actions, rewards, next_states = zip(*trajectory)
        states = pt.stack(states)
        actions = pt.concat(actions) if pt.is_tensor(action) else pt.Tensor(actions)
        actions = actions.type(pt.int)
        rewards = pt.concat(rewards) if pt.is_tensor(reward) else pt.Tensor(rewards)
        _, _, _, last_next_state = trajectory[-1]
        if pt.is_tensor(next_state):
            if pt.is_tensor(last_next_state):
                next_states = pt.stack(next_states)
            else:
                next_states = pt.stack(next_states[:-1])

        return cls(states, actions, rewards, next_states)

    @property
    def rewards(self) -> pt.Tensor:
        return self.__rewards.view(self.__rewards.shape)

    @property
    def states(self) -> pt.Tensor:
        return self.__states.view(self.__states.shape)

    @property
    def actions(self) -> pt.Tensor:
        return self.__actions.view(self.__actions.shape)

    @property
    def next_states(self) -> t.Optional[pt.Tensor]:
        if self.__next_states is None:
            return None
        return self.__next_states.view(self.__next_states.shape)

    def __len__(self):
        return self.__n

    def __add__(self, _):
        raise NotImplementedError('Cannot modify a frozen Trajectory')

    def __radd__(self, _):
        raise NotImplementedError('Cannot modify a frozen Trajectory')

    def __setslice__(self, _, __, ___):
        raise NotImplementedError('Cannot modify a frozen Trajectory')

    def __getitem__(self, item: int | slice):
        if isinstance(item, int):
            if item < 0:
                item += self.__n
            return self.__transition(item)
        elif isinstance(item, slice):
            # return [self.__transition(i) for i in range(self.n)[item]]
            return [self.__transition(i) for i in self.__index[item]]
        else:
            raise TypeError(f"{item} cannot be used to index a Trajectory")

    def __iter__(self):
        return [self.__transition(i) for i in range(self.__n)]

    def __transition(self, i: int) -> t.SoftSARS:
        if 0 > i or i >= self.__n:
            raise IndexError(f"Trajectory index {i} is out of bounds for size {self.n}")

        state = self.__states[i]
        action = self.__actions[i].item() if len(self.__actions.shape) == 1 else self.__actions[i]
        reward = self.__rewards[i].item() if len(self.__rewards.shape) == 1 else self.__rewards[i]
        if self.__next_states is None or i >= self.__next_states.shape[0]:  # The last transition may lack a next state
            next_state = None
        else:
            next_state = self.__next_states[i]
        return state, action, reward, next_state
