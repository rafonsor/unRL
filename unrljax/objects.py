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
