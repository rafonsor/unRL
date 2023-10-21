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
from typing import *  # noqa

from jax import Array

IntArray = Array  # expected dtype == jnp.int32
FloatArray = Array  # expected dtype == jnp.float32

IntLike = IntArray | int
FloatLike = FloatArray | float

Reward = float

T = TypeVar('T')


class MappedFrozenSet(frozenset, Generic[T]):
    def __new__(cls, iterable: Optional[Iterable[T]] = ()):
        return frozenset.__new__(cls, iterable)

    def __init__(self, iterable: Optional[Iterable[T]] = ()):
        super().__init__()
        self.__id_map = {s.id: s for s in self}
        self.__name_map = {s.name: s for s in self}
        self.__repr_map = {s.representation: s for s in self}

    def by_representation(self, representation: Any) -> T:
        return self.__repr_map[representation]

    def by_name(self, name: str) -> T:
        return self.__name_map[name]

    def by_id(self, id_: int) -> T:
        return self.__id_map[id_]


del T
