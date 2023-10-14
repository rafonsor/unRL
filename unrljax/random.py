import os
import sys
import threading
import time
import typing as t
from jax import random


def stob(value: t.Optional[str]) -> bool:
    if value is not None and value.lower() in ['y', 'yes', 'true', '1']:
        return True
    return False


def stoi(value: t.Optional[str], default: int = -1) -> int:
    if value is not None:
        try:
            return int(value)
        except ValueError:
            pass
    return default


USE_MASTER_SEED: bool = stob(os.environ.get('UNRL_USE_MASTER_SEED'))
MASTER_SEED: int = stoi(os.environ.get('UNRL_MASTER_SEED'), 42)

PRNGCollections: t.Dict[str, random.PRNGKeyArray] = {}
_PRNGCollections_lock = threading.Lock()


def get_prng_key(prng_collection: str = __name__) -> random.PRNGKeyArray:
    with _PRNGCollections_lock:
        if prng_collection in PRNGCollections:
            _, key = random.split(PRNGCollections[prng_collection])
        else:
            if USE_MASTER_SEED:
                seed = MASTER_SEED * int(prng_collection.encode().hex(), base=16)
            else:
                seed = time.perf_counter_ns()
            key = random.PRNGKey(seed)
        PRNGCollections[prng_collection] = key
    return key


class PRNGHandler:
    PRNGCollections: t.Dict[str, random.PRNGKeyArray]
    _PRNGCollections_lock: threading.Lock

    make_key: t.Callable[[str], random.PRNGKeyArray]

    def __init__(self):
        self.PRNGCollections = {}
        self._PRNGCollections_lock = threading.Lock()

        if USE_MASTER_SEED:
            self.make_key = self.make_key_with_random_seed
        else:
            self.make_key = self.make_key_with_deterministic_seed

    @staticmethod
    def make_key_with_random_seed(prng_collection: str) -> random.PRNGKeyArray:
        return random.PRNGKey(time.perf_counter_ns())

    @staticmethod
    def make_key_with_deterministic_seed(prng_collection: str) -> random.PRNGKeyArray:
        return random.PRNGKey((MASTER_SEED * int(prng_collection.encode().hex(), base=16)) % sys.maxsize)

    def get_prng_key(self, prng_collection: str) -> random.PRNGKeyArray:
        with self._PRNGCollections_lock:
            if prng_collection in self.PRNGCollections:
                _, key = random.split(self.PRNGCollections[prng_collection])
            else:
                key = self.make_key(prng_collection)
            self.PRNGCollections[prng_collection] = key
        return key


class PRNGMixin:
    prng_collection: str
    prng_handler: PRNGHandler

    def prng_key(self, prng_collection: t.Optional[str] = None) -> random.PRNGKey:
        return self.prng_handler.get_prng_key(prng_collection or self.prng_collection)

    def __init__(self, default_prng_collection: t.Optional[str]):
        self.prng_collection = default_prng_collection
        self.prng_handler = PRNGHandler()
