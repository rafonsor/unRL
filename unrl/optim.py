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
from torch.optim.optimizer import _use_grad_for_differentiable, params_t

import unrl.types as t
from unrl.config import validate_config


def _use_grad_for_differentiable_functional(differentiable: bool):
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


@_use_grad_for_differentiable_functional(differentiable=False)
def optimiser_update(model: pt.nn.Module, step_and_magnitude: t.FloatLike):
    """Increment parameters in the direction of their gradients.

    All gradients are subsequently reset"""
    if pt.is_tensor(step_and_magnitude):
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


class EligibilityTraceOptimizer(pt.optim.Optimizer):
    """Policy Gradient Optimizer with eligibility traces"""

    def __init__(self,
                 params: params_t,
                 discount_factor: float,
                 learning_rate: float,
                 trace_decay: float,
                 *,
                 weight_decay: float = 0.0,
                 discounted_gradient: bool = False):
        validate_config(discount_factor, "discount_factor", "unitpositive")
        validate_config(learning_rate, "learning_rate", "positive")
        validate_config(trace_decay, "trace_decay", "unit")
        validate_config(weight_decay, "weight_decay", "unit")
        defaults = {
            "discount_factor": discount_factor,
            "learning_rate": learning_rate,
            "trace_decay": trace_decay,
            "weight_decay": weight_decay,
            "discounted_gradient": discounted_gradient,
            "differentiable": False,
        }
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('differentiable', False)

    def _init_group(
        self,
        group: t.Dict,
        params_with_grad,
        grads,
        eligibility_traces,
        state_steps,  # Keep track of episode steps here? to apply discount_factor**step. Need to find way to reset step
    ):
        for p in group['params']:
            if p.grad is not None:
                params_with_grad.append(p)
                if p.grad.is_sparse:
                    raise RuntimeError('EligibilityTraceOptimizer does not support sparse gradients')
                grads.append(p.grad)

                state = self.state[p]
                if not state:
                    state['step'] = pt.tensor(0.)
                    state['eligibility_trace'] = pt.zeros_like(p)

                eligibility_traces.append(state['eligibility_trace'])
                state_steps.append(state['step'])

    @_use_grad_for_differentiable  # noqa
    def step(self, closure: t.Optional[t.Callable[[], float]] = None) -> t.Optional[float]:
        """Performs a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        assert closure is not None, f"{self.__class__.__name__}.step must receive a closure that provides a TD(0)-error"
        self._cuda_graph_capture_health_check()

        with pt.enable_grad():
            loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            eligibility_traces = []
            state_steps = []

            self._init_group(
                group,
                params_with_grad,
                grads,
                eligibility_traces,
                state_steps)

            self._eligibility_trace_update(
                loss,
                params_with_grad,
                grads,
                eligibility_traces,
                state_steps,
                lr=group['learning_rate'],
                y=group['discount_factor'],
                lam=group['trace_decay'],
                weight_decay=group['weight_decay'],
                discounted_gradient=group["discounted_gradient"]
            )

        return loss

    @staticmethod
    def _eligibility_trace_update(td: float,
                                  params_with_grad: t.TensorList,
                                  grads: t.TensorList,
                                  eligibility_traces: t.TensorList,
                                  state_steps: t.TensorList,
                                  lr: float,
                                  y: float,
                                  lam: float,
                                  weight_decay: float,
                                  discounted_gradient: bool):
        for param, grad, trace, step in zip(params_with_grad, grads, eligibility_traces, state_steps):
            step += 1
            trace.mul_(y*lam)
            if discounted_gradient:
                trace.add_(grad * (y ** (step - 1)))
            else:
                trace.add_(grad)

            d_p = trace*td
            if weight_decay:
                d_p -= weight_decay * param
            param.add_(d_p, alpha=lr)

    def episode_reset(self):
        self.state.clear()  # defaultdict, will repopulate parameter keys on-the-fly
