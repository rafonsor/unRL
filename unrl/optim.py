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
from functools import update_wrapper

import torch as pt

import unrl.types as t


def _use_grad_for_differentiable(differentiable: bool):
    """Adaptation of `torch.optim.optimizer::_use_grad_for_differentiable` for lone functions"""
    def decorator(fn):
        def _use_grad(*args, **kwargs):
            import torch._dynamo
            prev_grad = torch.is_grad_enabled()
            try:
                torch.set_grad_enabled(differentiable)
                torch._dynamo.graph_break()
                ret = fn(*args, **kwargs)
            finally:
                torch._dynamo.graph_break()
                torch.set_grad_enabled(prev_grad)
            return ret
        update_wrapper(_use_grad, fn)
        return _use_grad
    return decorator


@_use_grad_for_differentiable(differentiable=False)
def optimiser_update(model: pt.nn.Module, step_and_magnitude: t.FloatLike):
    """Increment parameters in the direction of their gradients.

    All gradients are subsequently reset"""
    if isinstance(step_and_magnitude, pt.Tensor):
        step_and_magnitude = step_and_magnitude.item()
    for p in model.parameters():
        if p.requires_grad:
            assert p.grad is not None, f"Attempting to perform gradient ascent on {p} with empty gradients"
            p.add_(p.grad, alpha=step_and_magnitude)
    model.zero_grad()


def optimiser_update_descent(model: pt.nn.Module, step_and_magnitude: t.FloatLike):
    """Decrement parameters in the direction of their gradients.

    Shorthand for calling `optimiser_update_ascent` with a negative step"""
    optimiser_update(model, -step_and_magnitude)
