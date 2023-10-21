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
from dataclasses import dataclass

import unrljax.types as t


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
    reward: t.Reward

    def __iter__(self) -> t.List[State | Action | t.Reward]:
        yield self.state
        yield self.action
        yield self.reward


@dataclass
class SARS(SAR):
    successor: State

    def __iter__(self) -> t.List[State | Action | t.Reward]:
        yield from super().__iter__()
        yield self.successor


Trajectory = t.Sequence[t.Tuple[State, Action, t.Reward] | SAR | SARS]

DiscreteStateSet = t.MappedFrozenSet[State]
DiscreteActionSet = t.MappedFrozenSet[Action]
