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
import math
import random

import unrl.types as t


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
