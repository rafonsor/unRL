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
from collections import deque
from threading import Lock

import torch as pt

import unrl.types as t
from unrl.config import validate_config
from unrl.containers import ContextualTransition


class ExperienceBufferProtocol(t.Protocol):
    def sample(self, n: int) -> t.SARST | t.Dict[str, pt.Tensor]:
        ...

    def append(self, transition: t.SARST | ContextualTransition, *args):
        ...


class ExperienceBuffer(ExperienceBufferProtocol, deque):
    def sample(self, n: int) -> t.SARST | t.Dict[str, pt.Tensor]:
        """Sample one or more experienced state transitions of the SARST form (with a termination flag relative to the
        next state). Sampling is with replacement.

        Args:
            n: number of transitions to retrieve.

        Raises:
            RuntimeError: attempting to sample from an empty buffer.

        Returns: a SARST tuple when a single sample is requested. Otherwise, transitions are merged into a dictionary
                 of stacked tensors for each type of transition value.
        """
        if len(self) == 0:
            raise RuntimeError("Cannot sample from an empty ExperienceBuffer")
        indices = pt.randint(0, len(self), (n,))
        if n == 1:
            return self[indices.item()]
        batch = [self[idx.item()] for idx in indices]
        states, actions, rewards, next_states, terminations = zip(*batch)
        return {
            "states": pt.stack(states),
            "actions": pt.stack(actions)[:, None],  # Expand to shape (BatchSize, 1)
            "rewards": pt.Tensor(rewards),
            "next_states": pt.stack(next_states),
            "terminations": pt.Tensor(terminations),
        }


class NaivePrioritisedExperienceBuffer(ExperienceBufferProtocol):
    """Naive implementation of the Prioritised Experience buffer that samples from all stored transitions in function of
    their relative priority. As explained in [1]_, this naive approach is inefficient for large buffer capacity.

    The public API is multi-threading safe.

    Attributes:
        maxlen: buffer capacity
        _indices: track position of elements stored
        _transitions: list of experienced transitions positioned following `_indices`
        _priorities: unidimensional tensor of priorities for transitions mapped according to `_indices`

    References:
        [1] Schaul, T., Quan, J., Antonoglou, I., & et al. (2015). "Prioritized experience replay". arXiv:1511.05952.
    """
    maxlen: int
    _indices: deque
    _transitions: t.List[t.SARST]
    _priorities: pt.Tensor
    __latest: int
    __mutex: Lock

    def __init__(self, maxlen: int):
        validate_config(maxlen, 'maxlen', 'positive')
        self.maxlen = maxlen
        self._indices = deque(maxlen=maxlen)
        # Pre-allocate capacity on internal containers
        self._transitions = [None] * maxlen
        self._priorities = pt.zeros((maxlen,), dtype=pt.float)
        self.__latest = -1  # So that the first retrieved index is "0"
        self.__mutex = Lock()

    def __add(self) -> int:
        """Retrieve and store the next usable index for storing prioritised transitions."""
        with self.__mutex:
            idx = (self.__latest + 1) % self.maxlen
            self.__latest = idx
            self._indices.append(idx)
        return idx

    def append(self, transition: t.SARST | ContextualTransition, priority: t.FloatLike):
        """Store a new transition to be prioritised according to its relative priority.

        Args:
            priority: unnormalised priority value. Note, this value should already be exponentiated.
            transition: experienced SARST transition to store.
        """
        idx = self.__add()
        self._transitions[idx] = transition
        self._priorities[idx] = priority

    def sample(self, n: int) -> t.SARST | t.Dict[str, pt.Tensor]:
        """Sample one or more experienced state transitions of the SARST form (with a flag indicating whether the
        resulting next state is terminal). Sampling is with replacement and according to a probability distribution of
        priorities.

        Args:
            n: number of transitions to retrieve.

        Raises:
            RuntimeError: attempting to sample from an empty buffer.

        Returns: a SARST tuple when a single sample is requested. Otherwise, transitions are merged into a dictionary
                 of stacked tensors for each type of transition value.
        """
        if len(self._indices) == 0:
            raise RuntimeError(f"Cannot sample from an empty {self.__class__.__name__}")
        with self.__mutex:
            batch = [
                self._transitions[loc]
                for loc
                in pt.distributions.Categorical(probs=self._priorities/self._priorities.sum()).sample((n,))  # noqa
            ]
        if n == 1:
            return batch[0]
        states, actions, rewards, next_states, terminations = zip(*batch)
        return {
            "states": pt.stack(states),
            "actions": pt.stack(actions)[:, None],  # Expand to shape (BatchSize, 1)
            "rewards": pt.Tensor(rewards),
            "next_states": pt.stack(next_states),
            "terminations": pt.Tensor(terminations),
        }
