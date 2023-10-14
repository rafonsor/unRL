import math
import random
import typing as t


def argmax(seq: t.Sequence) -> int:
    max_ = -math.inf
    arg = -1
    for i, value in enumerate(seq):
        if value > max_:
            max_ = value
            arg = i
    return arg


def argmax_all(seq: t.Sequence) -> t.List[int]:
    max_ = -math.inf
    args = []
    for i, value in enumerate(seq):
        if value > max_:
            max_ = value
            args = [i]
        elif value == max_:
            args.append(i)
    return args


def argmax_random(seq: t.Sequence) -> int:
    if seq:
        return random.choice(argmax_all(seq))
    return -1


def sample_gaussian(mu: float, sigma: float) -> float:
    """Gaussian sampling using the Box-Muller transform"""
    u1, u2 = random.random(), random.random()
    z = math.sqrt(-2*math.log(u1)) * math.cos(2*math.pi*u2)
    return mu + z*sigma
