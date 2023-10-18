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
