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
from typing import *

from numpy.typing import NDArray  # noqa
import torch as pt

TensorList: TypeAlias = List[pt.Tensor]
IntLike: TypeAlias = int | pt.IntType | pt.IntTensor | pt.Tensor
FloatLike: TypeAlias = float | pt.FloatType | pt.FloatTensor | pt.Tensor
BoolLike: TypeAlias = bool | pt.BoolType | pt.BoolTensor | pt.Tensor

# State-Action-Reward-NextState tuples
SoftSARS: TypeAlias = Tuple[pt.Tensor, IntLike, FloatLike, Optional[pt.Tensor]]  # state is terminal when no next state
SARST: TypeAlias = Tuple[pt.Tensor, IntLike, FloatLike, pt.Tensor, bool]  # Indicates whether next state is terminal
