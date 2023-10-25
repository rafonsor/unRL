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

# Validators
def validate_config(value: int | float, label: str, validator: str):
    """Check `value` set for configuration `label` satisfies requirements  of `validator`. If not, an exception is
    raised.

    Args:
        value: value to check
        label: name of configuration
        validator: type of value check to apply

    Raises:
        ValueError: value fails validation
        NotImplementedError: validation unsupported
    """
    if validator == "unit":
        validate_unit_config(value, label)
    elif validator == "unitpositive":
        validate_unitpositive_config(value, label)
    elif validator == "positive":
        validate_positive_config(value, label)
    elif validator == "nonnegative":
        validate_nonnegative_config(value, label)
    elif validator == "negative":
        validate_negative_config(value, label)
    else:
        raise NotImplementedError(f'Unsupported validator "{validator}" requested for config {label}.')


def validate_unit_config(value: int | float, label: str):
    if not 0 <= value <= 1:
        raise ValueError(f'Config "{label}" not bounded to [0, 1].')


def validate_unitpositive_config(value: int | float, label: str):
    if not 0 < value <= 1:
        raise ValueError(f'Config "{label}" not bounded to (0, 1].')


def validate_positive_config(value: int | float, label: str):
    if not 0 < value:
        raise ValueError(f'Config "{label}" is not strictly positive.')


def validate_nonnegative_config(value: int | float, label: str):
    if not 0 <= value:
        raise ValueError(f'Config "{label}" is negative.')


def validate_negative_config(value: int | float, label: str):
    if 0 <= value:
        raise ValueError(f'Config "{label}" is not strictly negative.')
