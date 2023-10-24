# NOTE: this is a work in progress, feedback and contributions are welcome.

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

"""Module docstring is used to summarise the purpose and core functionality provided by the module at hand. Must follow
the copyright notice.

The overall docstyle chosen by unRL is a lax adaptation of google's. Note that mixed indentation is not desirable.

Examples:
    Instructions on how to use public APIs or functions can be included within the module docstring.
"""

# Imports: sorted by module path in relative alphabetical order, following the 4 groups defined below. Whole package
# imports must precede partial imports. Deviate from this structure when appropriate according to common practices.
# However, `unrl.types` must be used as the source of built-in and custom types instead of `typing` directly.

# Future imports
from __future__ import annotations

# Standard Library imports
import os.path
from abc import abstractmethod, ABCMeta
from enum import Enum

# Third-party imports
# Use the standard abbreviations as defined below. All other packages, modules, objects should not be abbreviated unless
# to prevent implicit redeclaration.
import gymnasium as gym
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb
import torch as pt
import torch.nn.functional as F
from jax import random
from torch import nn

# unRL imports
import unrljax.types as t


T = t.TypeVar('T')


# Note: instructions that follow for class methods are equally applicable to named functions. Non-ephemeral local
# functions, that is, local functions dynamically constructed to decorate an object or meant for non-local used, must
# abide by the same documentation guidelines.

class ExampleClass:
    """Class docstrings should start with a brief description.

    Strive to include explanations for any unintuitive logic, sources for the implementation (see "References" below),
    or design choices. The latter is only warranted when differing from the common practices followed in unRL.

    Describe class and public instance attributes. In terms of type descriptions, type annotations are mandatory, do not
    repeat them in docstrings except when used to provide additional clarifications in ambiguous cases.

    Attributes:
        model: Instance attribute

    References must be specified within the class docstring and should follow the APA style as exemplified below. Where
    appropriate cite such references inside docstrings using `[#]_`. Write at most 3 authors, resort to `et al.` beyond
    this.

    References:
        [1] Lapicque, L. (1906). "The Electrical Excitation Of Nerves And Muscles". The British Medical Journal.
        [2] McCulloch, W. S., & Pitts, W. (1943). A logical calculus of the ideas immanent in nervous activity. The
            bulletin of mathematical biophysics, 5, pp. 115-133.
        [#] <Last Name>, <Initials> (<Year>). [Section|Chapter> <#[-#]>]"<Book Title Respecting the Title Case Style>".
            <Publisher>.
        [#] <Last Name>, <Initials>, & <Last Name>, <Initials>, & <Last Name>, <Initials>, & et al. (<Year>). "<Document
            Title Respecting the Title Case Style>". <Publication>[, <volume>][, pp. <FirstPage#-LastPage#].
    """

    # Specify all public instance attributes and include type annotations
    model: t.Any

    def __init__(self, model_factory: t.Callable[..., t.Any], **model_specs: t.Dict[str, t.Any]):
        """__ini__ should leave this line empty and only include Args
        Args:
            model_factory: description should not repeat the type
            **model_specs: rename "kwargs" when it such keyword arguments pertain to a specific concept
        Raises:
            RuntimeError: specify all exceptions explicitly thrown by the method or function at hand
        """
        assert callable(model_factory), "Assertions can be used for input validation, before any operation is applied"
        try:
            self.model = model_factory(**model_specs)
        except ValueError as ex:  # Abbreviate exceptions caught as `ex`.
            raise RuntimeError('How not to use a factory pattern') from ex

    def public_method_without_return(self):
        """Do not add a return type annotation or docstring description in methods and functions without a return value
        (implicit None). Use plain "return" statements to terminate the execution flow earlier."""
        ...

    def public_method_without_optional(self, seq: t.Sequence[T]) -> t.Optional[T | t.Array]:
        """Use typing.Optional whenever a method or function does not guarantee a return value, in which case explicit
        "return None" statements are needed. Use the "Returns" heading to describe the conditions for each return type,
        including the None case, when not self-evident.

        Args:
            seq: Some sequence to process

        Returns:
            the expanded input sequence. If `seq` only contains 1 element, then a loose instance of T is returned. When
            `seq` is empty, None is returned.
        """
        ...


del T  # Prune temporary variables when no longer necessary
