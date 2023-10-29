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
from unrl.utils import optional_lock

IndexedSARSTBatch: t.TypeAlias = t.Tuple[t.Dict[str, pt.Tensor], t.Dict[str, pt.Tensor]]


def _extract_batch_from_indices(container: t.Sequence[t.SARST | ContextualTransition],
                                indices: pt.Tensor,
                                /,
                                container_mutex: t.Optional[Lock] = None
                                ) -> t.Dict[str, pt.Tensor]:
    """Extract and process a subset of transitions.

    Args:
        container: indexable container with SARST transitions
        indices: position of elements to retrieve from `container`
    Keyword Args:
        container_mutex: (Optional) an optional lock to acquire when retrieving elements from `container`

    Returns:
        A collection of SARST transitions that decomposes and groups their constituent elements (e.g. states, rewards)
        into stacked tensors.
    """
    with optional_lock(container_mutex):
        batch = [container[idx] for idx in indices]
    states, actions, rewards, next_states, terminations = zip(*batch)
    return {
        "states": pt.stack(states),
        "actions": pt.stack(actions)[:, None],  # Expand to shape (BatchSize, 1)
        "rewards": pt.Tensor(rewards),
        "next_states": pt.stack(next_states),
        "terminations": pt.Tensor(terminations),
    }


class ExperienceBufferProtocol(t.Protocol):
    def sample(self, n: int) -> IndexedSARSTBatch:
        """Sample one or more experienced state transitions of the SARST form (with a termination flag relative to the
        next state). Sampling is with replacement.

        Args:
            n: number of transitions to retrieve.

        Raises:
            RuntimeError: attempting to sample from an empty buffer.

        Returns:
            A tuple (batch, sampling metadata) containing a collection of SARST transitions and a metadata dictionary
            from the sampling process that must include the sampled indices. Transitions are decomposed by their
            constituent elements (e.g. states, rewards) and stacked as tensors.
        """
        ...

    def append(self, transition: t.SARST | ContextualTransition, *args):
        """Store new transition to buffer

        Args:
            transition: transition of the SARST form.
            *args: (Optional) extra arguments needed by specific buffer implementations.
        """
        ...

    def __len__(self) -> int:
        ...


class ExperienceBuffer(deque, ExperienceBufferProtocol):
    def sample(self, n: int) -> IndexedSARSTBatch:
        """Sample one or more experienced state transitions of the SARST form (with a termination flag relative to the
        next state). Sampling is with replacement.

        Args:
            n: number of transitions to retrieve.

        Raises:
            RuntimeError: attempting to sample from an empty buffer.

        Returns:
            A tuple (batch, sampling metadata) containing a collection of SARST transitions and a metadata dictionary
            from the sampling process that includes the sampled indices. Transitions are merged into a dictionary of
            stacked tensors for each type of transition value.
        """
        if len(self) == 0:
            raise RuntimeError("Cannot sample from an empty ExperienceBuffer")
        indices = pt.randint(0, len(self), (n,))
        batch = _extract_batch_from_indices(self, indices)
        # states, actions, rewards, next_states, terminations = zip(*[self[idx.item()] for idx in indices])
        # batch = {
        #     "states": pt.stack(states),
        #     "actions": pt.stack(actions)[:, None],  # Expand to shape (BatchSize, 1)
        #     "rewards": pt.Tensor(rewards),
        #     "next_states": pt.stack(next_states),
        #     "terminations": pt.Tensor(terminations),
        # }
        return batch, {"indices": indices}


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
    _mutex: Lock
    __latest: int

    def __init__(self, maxlen: int, *, epsilon: float = 1e-8):
        """Args:
            maxlen: buffer capacity
            epsilon: small epsilon added to all priorities to ensure non-zero sampling probability for any stored
                     transitions.
        """
        validate_config(maxlen, 'maxlen', 'positive')
        self.maxlen = maxlen
        self._indices = deque(maxlen=maxlen)
        # Pre-allocate capacity on internal containers
        self._transitions = [None] * maxlen
        self._priorities = pt.zeros((maxlen,), dtype=pt.float)
        self._eps = epsilon
        self.__latest = -1  # So that the first retrieved index is "0"
        self._mutex = Lock()

    def __len__(self):
        return len(self._indices)

    def __add(self) -> int:
        """Retrieve and store the next usable index for storing prioritised transitions."""
        with self._mutex:
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
        self._priorities[idx] = priority + self._eps

    def sample(self, n: int) -> IndexedSARSTBatch:
        """Sample one or more experienced state transitions of the SARST form (with a flag indicating whether the
        resulting next state is terminal). Sampling is with replacement and according to a probability distribution of
        priorities assigned to each stored transition.

        Args:
            n: number of transitions to retrieve.

        Raises:
            RuntimeError: attempting to sample from an empty buffer.

        Returns:
            A tuple (batch, sampling metadata) containing a collection of SARST transitions and a metadata dictionary
            from the sampling process that includes the sampled indices and their probabilities. Transitions are merged
            into a dictionary of stacked tensors for each type of transition value.
        """
        if len(self._indices) == 0:
            raise RuntimeError(f"Cannot sample from an empty {self.__class__.__name__}")
        with self._mutex:
            probs = self._priorities/self._priorities.sum()
            indices = pt.distributions.Categorical(probs=probs).sample((n,))  # noqa
        batch = _extract_batch_from_indices(self._transitions, indices, container_mutex=self._mutex)
        metadata = {"indices": indices, "probabilities": probs[indices]}
        return batch, metadata

    def set_priority(self, indices: pt.Tensor, priorities: pt.Tensor):
        """Update priority values of existing transitions.

        Args:
            indices: transitions to re-prioritise, identified by the index included in a previous sampling.
            priorities: new priority values.
        """
        assert indices.shape == priorities.shape, \
            f"Inputs dimensions mismatch ({indices.shape} and {priorities.shape}), cannot update priority values."
        with self._mutex:
            self._priorities[indices] = priorities


class RankPrioritisedExperienceBuffer(NaivePrioritisedExperienceBuffer):
    """Naive rank-based variant of the Prioritised Experience buffer. The probability distribution is
    constructed from the inverse of the relative ranks of priorities. THe complexity of this approach is proportional to
    the buffer capacity.

    References:
        [1] Schaul, T., Quan, J., Antonoglou, I., & et al. (2015). "Prioritized experience replay". arXiv:1511.05952.
    """
    def __init__(self, maxlen: int, *, epsilon: float = 1e-8):
        """Args:
            maxlen: buffer capacity
            epsilon: small epsilon added to all priorities to ensure non-zero sampling probability for any stored
                     transitions.
        """
        super().__init__(maxlen, epsilon=epsilon)
        self.__probs = 1 / pt.arange(1, self.maxlen + 1)

    def sample(self, n: int) -> IndexedSARSTBatch:
        """Sample one or more experienced state transitions of the SARST form (with a flag indicating whether the
        resulting next state is terminal). Sampling is with replacement and according to a probability distribution of
        ranked priorities assigned to each stored transition.

        Args:
            n: number of transitions to retrieve.

        Raises:
            RuntimeError: attempting to sample from an empty buffer.

        Returns:
            A tuple (batch, sampling metadata) containing a collection of SARST transitions and a metadata dictionary
            from the sampling process that includes the sampled indices and their probabilities. Transitions are merged
            into a dictionary of stacked tensors for each type of transition value.
        """
        if len(self._indices) == 0:
            raise RuntimeError(f"Cannot sample from an empty {self.__class__.__name__}")
        with self._mutex:
            # Since we're interested in the arguments, and not the actual priority values, we sample from the unordered
            # probability distribution P where ``p(i) = 1/rank(i)``.
            rank_index = pt.argsort(self._priorities, descending=True)
            ranks = pt.distributions.Categorical(probs=self.__probs[:len(self)]).sample((n,))  # noqa: t.Tuple[int] accepted
            indices = pt.take_along_dim(rank_index, ranks, dim=-1)
        batch = _extract_batch_from_indices(self._transitions, indices, container_mutex=self._mutex)
        metadata = {"indices": indices, "probabilities": self.__probs[indices].detach()}
        return batch, metadata


class RankPartitionedPrioritisedExperienceBuffer(NaivePrioritisedExperienceBuffer):
    """Sampling-efficient Rank-based variant of the Prioritised Experience buffer proposed by [1]_. Ranked priorities
    are partitioned into `k` partitions. Sampling occurs then uniformly within each partition.

    References:
        [1] Schaul, T., Quan, J., Antonoglou, I., & et al. (2015). "Prioritized experience replay". arXiv:1511.05952.
    """
    def __init__(self, maxlen: int, *, partitions: int, epsilon: float = 1e-8):
        """Args:
            maxlen: buffer capacity
            epsilon: small epsilon added to all priorities to ensure non-zero sampling probability for any stored
                     transitions.
        """
        super().__init__(maxlen, epsilon=epsilon)
        validate_config(partitions, 'partitions', 'positive')
        self.k = partitions
        self._partition_starts = None
        self._partition_length = 0
        self.__probs = 1 / pt.arange(1, self.maxlen + 1)

    def _update_partitions(self):
        n = len(self._indices)
        length = max(n // self.k, 1)
        partitions = (pt.arange(0, self.k) * length).clip(max=n-length)
        self._partition_starts = partitions
        self._partition_length = length

    def append(self, transition: t.SARST | ContextualTransition, priority: t.FloatLike):
        """Store a new transition to be prioritised according to its relative priority rank.

        Args:
            priority: unnormalised priority value. Note, this value should already be exponentiated.
            transition: experienced SARST transition to store.
        """
        super().append(transition, priority)
        self._update_partitions()

    def sample(self, n: int) -> IndexedSARSTBatch:
        """Sample one or more experienced state transitions of the SARST form (with a flag indicating whether the
        resulting next state is terminal). Uses a dual uniform-sampling process with replacement: first of partitions
        and subsequently of transitions within the chosen partitions of ranked priorities.

        Note, as in [1]_, when the requested batch size `n` equal the number of partitions then sampling is stratified:
        exactly one transition is sample from each partition.

        Args:
            n: number of transitions to retrieve.

        Raises:
            RuntimeError: attempting to sample from an empty buffer.

        Returns:
            A tuple (batch, sampling metadata) containing a collection of SARST transitions and a metadata dictionary
            from the sampling process. Transitions are merged into a dictionary of stacked tensors for each type of
            transition value. Metadata contains the sampled indices, their probabilities, and the partitions each
            belongs to.
        """
        if len(self._indices) == 0:
            raise RuntimeError(f"Cannot sample from an empty {self.__class__.__name__}")
        # Select partitions to sample transitions from
        if n == self.k:
            # When the batch size `n` matches the number of partitions, all partitions are sampled from exactly once
            # like indicated in [1]_.
            partitions = pt.arange(0, self.k)
        else:
            partitions = pt.randint(0, self.k, (n,))
        with self._mutex:
            # Since we're interested in the arguments, and not the actual priority values, we sample from the unordered
            # probability distribution P where ``p(i) = 1/rank(i)``.
            ranks = pt.argsort(self._priorities, descending=True)
            indices = pt.take_along_dim(
                ranks,
                pt.randint(0, self._partition_length, (n,)) + self._partition_starts[partitions],
                dim=-1)
        batch = _extract_batch_from_indices(self._transitions, indices, container_mutex=self._mutex)
        metadata = {"indices": indices, "probabilities": self.__probs[indices].detach(), "partitions": partitions}
        return batch, metadata
