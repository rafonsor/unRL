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
import warnings
from dataclasses import dataclass
from functools import update_wrapper

import torch as pt
from torch.optim.optimizer import _use_grad_for_differentiable, params_t

import unrl.types as t
from unrl.config import validate_config
from unrl.utils import flat_tensor, moving_average_inplace


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
        self._cuda_graph_capture_health_check()

        if closure is not None:
            with pt.enable_grad():
                loss = closure()
        else:
            loss = 1

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


def multi_optimiser_stepper(*optimisers: pt.optim.Optimizer) -> t.Callable[[pt.Tensor, ...], None]:
    """Helper function to one-click backprogate and update parameters for multiple models sharing a common loss value"""
    def stepper(loss: pt.Tensor, **backward_kwargs) -> None:
        """Backpropagate loss tensor and update parameters using all injected optimisers.

        Args:
            loss: loss tensor used to generate gradients.
            **backward_kwargs: optional keyword arguments to pass to `.backward`.
        """
        for optim in optimisers:
            optim.zero_grad()
        loss.backward(**backward_kwargs)
        for optim in optimisers:
            optim.step()
    return stepper


def polyak_averaging_inplace(models: t.Sequence[pt.nn.Module], weights: t.Sequence[float]):
    """Inplace weighted averaging of model parameters

    Args:
        models: sequence of twin models from which to compute weighted averages. The parameters of the
                first model (0th index) are modified inplace.
        weights: importance of each model
    """
    assert models is not None and len(models) >= 2, 'Must provide at least two models'
    assert weights is not None and len(models) == len(weights), "Inputs dimensions mismatch"

    with pt.no_grad():
        for ps in zip(*(m.parameters() for m in models)):
            pmain = ps[0]
            pmain.set_(sum(p*w for p, w in zip(ps, weights)))


class KFACOptimizer(pt.optim.SGD):
    """Kronecker-factored Approximate Curvature Optimizer is a second-order natural gradient descent optimisation method
    that approximates the inverse of the Fisher Information Matrix with Kronecker factors layer-by-layer (K-FAC, [1]_).

    This optimizer implements both block-diagonal and block-tridiagonal approximations of FIM as proposed in [1]_. Flag
    `block_tridiagional` controls which approximation method to use. Choosing either involves a trade-off between speed
    and accuracy.

    Note: Support for convolutional layers, proposed for example in [3]_, is deferred to future versions. Parameters in
    unsupported layers are updated according to gradient descent with momentum.

    Simplifications:
        - No additional backward and forward passes are performed on sampled subsets of the current minibatch, instead,
          statistics are kept about the entire minibatch.
        - Only a single candidate for factored Tikhonov damping strength is maintained, the unscaled "gamma" `(√lambda+eta)` recomputed
          every `factored_tikhonov_refresh_steps` (known as T2 in [1]_).

    References:
        [1] Martens, J., & Grosse, R. (2015). "Optimizing neural networks with kronecker-factored approximate curvature"
            . In Proceedings of the 32nd International Conference on Machine Learning.
        [2] Ba, J., Grosse, R., & Martens, J. (2016). "Distributed second-order optimization using Kronecker-factored
            approximations". In Proceedings of the 33rd International Conference on Machine Learning.
    """
    @dataclass(slots=True)
    class ModuleState:
        tikhonov_damping: float
        learning_rate: float
        momentum: float = 1.0
        rescaling_denominator: pt.Tensor = 1.0
        activation: pt.Tensor = None
        grad_output: pt.Tensor = None
        a_bar_inv: pt.Tensor = None
        g_inv: pt.Tensor = None
        final_update: pt.Tensor = 0.0

    model: pt.nn.Module
    handles: t.Dict[pt.nn.Module, t.Dict[str, "RemovableHandle"]]
    module_state: t.Dict[pt.nn.Module, ModuleState]

    def __init__(self,
                 model: pt.nn.Module,
                 trust_region_radius: float,
                 max_epsilon: float,
                 max_learning_rate: float,
                 learning_rate: float,
                 momentum: float = 0,
                 dampening: float = 0,
                 weight_decay: float = 1e-5,
                 max_epsilon: float = 0.95,
                 tikhonov_damping: float = 150.0,
                 tikhonov_refresh_steps: int = 5,
                 factored_tikhonov_refresh_steps: int = 20,
                 inverse_refresh_steps: int = 20,
                 alpha_mu_refresh_steps: int = 5,
                 block_tridiagional: bool = False,
                 tikhonov_scaling: t.Optional[float] = None):
        """
        Args:
            model: K-FAC compatible model to optimize.
            trust_region_radius: boundary in the Trust Region sphere to clip learning rates.
            max_epsilon: maximum rate to use when smoothing activations and loss gradients.
            max_learning_rate: maximum rate to allow when clipping learning rates.
            learning_rate: default learning rate for unsupported parameters.
            momentum: default momentum rate for unsupported parameters.
            dampening: default dampening rate for unsupported parameters.
            weight_decay: L2 regularisation coefficient (known as η).
            tikhonov_damping: initial strength of Tikhonov damping (known as λ).
            tikhonov_refresh_steps: Tikhonov damping rescale interval (known as T1).
            factored_tikhonov_refresh_steps: factored Tikhonov damping rescale interval (known as T2).
            inverse_refresh_steps: interval for computing inverses of outer products (known as T3).
            alpha_mu_refresh_steps: interval for computing optimal learning rates and momentum factors.
            block_tridiagional: use Block-Tridiagonal approximation instead of Block-Diagonal.
            tikhonov_scaling: (Optional) custom Tikhonov rescaling factor (known as w_1)
        """
        validate_config(learning_rate, "learning_rate", "unitpositive")
        validate_config(trust_region_radius, "trust_region_radius", "positive")
        validate_config(max_epsilon, "max_epsilon", "unit")
        validate_config(max_learning_rate, "max_learning_rate", "unitpositive")
        validate_config(momentum, "momentum", "unitp")
        validate_config(dampening, "dampening", "unit")
        validate_config(weight_decay, "weight_decay", "unit")
        validate_config(tikhonov_damping, "tikhonov_damping", "positive")
        validate_config(tikhonov_refresh_steps, "tikhonov_refresh_steps", "positive")
        validate_config(factored_tikhonov_refresh_steps, "factored_tikhonov_refresh_steps", "positive")
        validate_config(inverse_refresh_steps, "inverse_refresh_steps", "positive")
        validate_config(alpha_mu_refresh_steps, "alpha_mu_refresh_steps", "positive")
        if tikhonov_scaling is not None:
            validate_config(tikhonov_scaling, "tikhonov_scaling", "unit")

        self.trust_region_radius = trust_region_radius
        self.max_epsilon = max_epsilon
        self.max_learning_rate = max_learning_rate
        self.weight_decay = weight_decay
        if block_tridiagional:
            warnings.warn(
                UserWarning("K-FAC with Block-Tridiagonal approximation is not yet available, using Block-Diagonal."))
            block_tridiagional = False
        self.block_tridiagional = block_tridiagional

        # Update intervals
        # T1: updating "lambda", the Tikhonov damping by rescaling with `tikhonov_damping` (w_1 in [1]_)
        self.tikhonov_refresh_steps = tikhonov_refresh_steps
        self.tikhonov_damping = tikhonov_damping
        self.tikhonov_scaling = tikhonov_scaling or (19/20) ** self.tikhonov_refresh_steps
        # T2: updating "gamma", the strength of factored Tikhonov damping
        self.factored_tikhonov_refresh_steps = factored_tikhonov_refresh_steps
        self.factored_tikhonov_damping = (self.tikhonov_damping * self.weight_decay) ** 0.5
        # T3: updating ~F^{-1}, the approximated FIM inverse
        self.inverse_refresh_steps = inverse_refresh_steps
        # T4: New period introduced to control the computation of the optimal learning rate (α) and momentum (µ)
        self.alpha_mu_refresh_steps = alpha_mu_refresh_steps

        self.model = model
        self.handles = {}
        self.module_state = {}
        self._steps = 0

        self.toggle()
        self._reset_modules_state()

        self.module_params, supported_params, unsupported_params = self._gather_parameters()
        assert len(supported_params), f"No module found in {type(model).__name__} is compatible for K-FAC optimization."
        super().__init__(supported_params, lr=1.0)  # K-FAC's learning rate scaling and momentum are handled outside

        if len(unsupported_params):
            # Unsupported modules are updated via gradient descent with momentum
            self._fallback_sgd = pt.optim.SGD(
                supported_params, lr=learning_rate, momentum=momentum, dampening=dampening, weight_decay=weight_decay)
        else:
            self._fallback_sgd = None

    def __setstate__(self, state):
        super().__setstate__(state)
        self._reset_modules_state()

    def __del__(self):
        self._deregister_model(self.handles)

    def _reset_modules_state(self):
        self.module_state = {
            mod: KFACOptimizer.ModuleState(tikhonov_damping=self.tikhonov_damping, learning_rate=self.max_learning_rate)
            for mod in self.handles
        }

    @staticmethod
    def _register_model(model: pt.nn.Module,
                        forward_hook: callable,
                        backward_hook: callable
                        ) -> t.Dict[pt.nn.Module, t.Dict[str, "RemovableHandle"]]:
        warnings.warn(RuntimeWarning("Only Linear layers are currently supported, other modules are updated using GD."))
        return {
            mod: {
                "forward": mod.register_forward_hook(forward_hook),
                "backward": mod.register_backward_hook(backward_hook)
            }
            for mod in model.modules()
            if isinstance(mod, pt.nn.Linear)
        }

    @staticmethod
    def _deregister_model(handles: t.Dict[pt.nn.Module, t.Dict[str, "RemovableHandle"]]):
        [rh.remove() for hooks in handles.values() for rh in hooks.values()]
        handles.clear()

    def _gather_parameters(
            self
    ) -> t.Tuple[t.Dict[pt.nn.Module, pt.nn.Parameter], t.List[pt.nn.Parameter], t.List[pt.nn.Parameter]]:
        """Gather all the parameters, discriminated by their compatibility with K-FAC optimisation.

        Returns:
            Tuple (module-parameters map, supported, unsupported) containing first a dictionary of gradient-aware
            parameters for K-FAC compatible modules and two lists of parameters to manage with K-FAC and SGD optimisers.
        """
        module_params_map = {
            mod: p
            for mod in self.model.modules()
            for p in mod.parameters(recurse=False)
            if mod in self.handles and p.requires_grad
        }
        supported = [p for params in module_params_map.values() for p in params]
        unsupported = [
            p
            for mod in self.model.modules()
            for p in mod.parameters(recurse=False)
            if mod not in self.handles
        ]
        return module_params_map, supported, unsupported

    def _forward_hook(self, module, args, output) -> None:
        inputs = flat_tensor(args)
        state = self.module_state[module]
        if state.activation is None:
            state.activation = inputs
        else:
            moving_average_inplace(state.activation, inputs, self.epsilon)

    def _backward_hook(self, module, grad_input, grad_output) -> None:
        grad_output = grad_output.reshape(-1)
        state = self.module_state[module]
        if state.grad_output is None:
            state.grad_output = grad_output
        else:
            moving_average_inplace(state.grad_output, grad_output, self.epsilon)

    @property
    def epsilon(self) -> float:
        """Exponentially decaying averaging rate based on the current number of steps (Sect.5 [1]_)"""
        return min(1 - 1/self._steps, self.max_epsilon)

    def toggle(self):
        if self.handles:
            self._deregister_model(self.handles)
        else:
            self.handles = self._register_model(self.model, self._forward_hook, self._backward_hook)

    @staticmethod
    def eiginverse(tensor: pt.Tensor) -> t.Tuple[pt.Tensor, pt.Tensor]:
        """Invert tensor using eigendecomposition. The average eigenvalue of `tensor` is also returned, since it serves
        as a proxy to its scaled trace (useful for applying Tikhonov damping).

        Args:
            tensor: Tensor to invert.

        Returns:
            A tuple (inverse, average eigenvalue) containing the inverse and average eigenvalue of `tensor`.
        """
        eigs = pt.linalg.eigh(tensor @ tensor.T)
        inv = eigs.eigenvectors @ (pt.eye(eigs.eigenvalues.shape)*eigs.eigenvalues).T @ eigs.eigenvectors.T
        avg = eigs.eigenvalues.mean()
        del eigs
        return inv, avg

    def _compute_forwards(self, state: ModuleState) -> t.Tuple[pt.Tensor, pt.Tensor]:
        """Compute the outer products of the smoothed activations and output gradients for the requested module
        and apply damping using the alternative Tikhonov technique of [1]_.

        Args:
            state: a model's module state to process

        Raises:
            AssertionError: when he provided module state does not yet contain the needed quantities.

        Returns:
            Tuple (act, dout) containing the dampened outer products of activations and output gradients, respectively.
            The approximate FIM can be obtained using their Kronecker product.
        """
        assert state.activation is not None and state.grad_output is not None, "Calling `step()` before initialisation."
        # Compute outer products
        act = state.activation @ state.activation.T
        dout = state.grad_output @ state.grad_output.T
        # Add damping
        act_scaled_trace = pt.trace(act) / act.shape[0]
        dout_scaled_trace = pt.trace(dout) / dout.shape[0]
        pi = pt.sqrt(act_scaled_trace / dout_scaled_trace)
        act += self.factored_tikhonov_damping / pi
        dout += self.factored_tikhonov_damping * pi
        return act, dout

    def _compute_inverses(self, state: ModuleState) -> t.Tuple[pt.Tensor, pt.Tensor]:
        """Compute the inverted outer products of the smoothed activations and output gradients for the requested module
        and apply damping using the alternative Tikhonov technique of [1]_.

        This function leverages eigendecomposition and orthogonality relations to compute inverses. The function `eigh`
        is used for eigendecomposition to ensure only real-valued matrices are returned.

        Args:
            state: a model's module state to process

        Raises:
            AssertionError: when he provided module state does not yet contain the needed quantities.

        Returns:
            A tuple (act_inv, dout_inv) containing the dampened inverse outer products of activations and output
            gradients, respectively.
        """
        assert state.activation is not None and state.grad_output is not None, "Calling `step()` before initialisation."
        # Compute the inverse of the outer products
        act_inv, act_scaled_trace = self.eiginverse(state.activation @ state.activation.T)
        dout_inv, dout_scaled_trace = self.eiginverse(state.grad_output @ state.grad_output.T)
        # Add damping
        pi = pt.sqrt(act_scaled_trace / dout_scaled_trace)
        act_inv += self.factored_tikhonov_damping / pi
        dout_inv += self.factored_tikhonov_damping * pi
        return act_inv, dout_inv

    def _update_inverses(self):
        if self._steps % self.inverse_refresh_steps == 0:
            for state in self.module_state.values():
                state.a_bar_inv, state.g_inv = self._compute_inverses(state)

    def _compute_rescaling_and_momentum(self, state: ModuleState, update_proposal: pt.Tensor) -> t.Tuple[float, float]:
        """
        Note, the optimal alpha still has to undergo a possible clipping to stay inside the Trust Region sphere.
        [α∗] = −[ ∆F ∆ + (λ + η)‖∆‖22    ∆>F δ0 + (λ + η)∆>δ0]−1     [∇h>∆]
        [μ∗] = -[ ∆>F δ0 + (λ + η)∆>δ0    δ>0 F δ0 + (λ + η)‖δ0‖22]    [∇h>δ]
        Abstracted as to `[α; μ]^{-1} = [x1 + y1, x2 + y2; x2 + y2, x3 + y3]^{-1} * [u; v]`
        """
        # ∇h: loss gradient
        # ∆: update proposal
        # δ_0: previous final update
        # λ: Tikhonov damping
        # η: weight decay (L2 regularization coefficient)
        # F: here just the module's forward approximation
        damping = state.tikhonov_damping + self.weight_decay
        y1 = damping * pt.inner(update_proposal, update_proposal)
        y2 = damping * pt.inner(update_proposal, state.final_update)
        y3 = damping * pt.inner(state.final_update, state.final_update)
        xy = pt.Tensor([[y1, y2], [y2, y3]])

        Fmod = pt.kron(*self._compute_forwards(state))
        x2 = update_proposal.T @ Fmod @ state.final_update
        xy[0, 0] += update_proposal.T @ Fmod @ update_proposal
        xy[0, 1] += x2
        xy[1, 0] += x2
        xy[1, 1] += state.final_update.T @ Fmod @ state.final_update

        uv = pt.Tensor([pt.inner(state.grad_output, update_proposal), pt.inner(state.grad_output, state.final_update)])
        optimal = pt.inverse(xy).dot(uv)
        return optimal[0], optimal[1]

    def _clip_learning_rate(self, grad_loss: pt.Tensor, update_proposal: pt.Tensor, learning_rate: float) -> float:
        """Clip learning rates in accordance to the Trust Region radius limit.

        Args:
            update_proposal: natural gradients wrt parameters considered.
            grad_loss: original loss gradients wrt parameters considered.
            learning_rate: most recent optimal alpha.

        Returns:
            learning rate clipped to within the Trust Region sphere.
        """
        # We use the inner product between update proposals and the original gradients (see Sect.5 [2]_).
        inner_product_sum = update_proposal.T.dot(grad_loss)
        max_learning_rate = min(self.max_learning_rate, pt.sqrt(self.trust_region_radius / inner_product_sum).item())
        return min(learning_rate, max_learning_rate)

    def _quadratic_cost_model(self, state: ModuleState, loss: pt.Tensor) -> pt.Tensor:
        """Produce a scalar approximation cost for a specific module's FIM block. This cost quantifies how well the
        update compensates the observed loss (wrt module in question), thereby indicating the quality of convergence.

        Args:
            state: state of the module being analysed.
            loss: model's loss.

        Returns:
            Tensor containing the scalar approximation cost.
        """
        Abar, G = self._compute_forwards(state)
        quadratic_term = 0.5*(state.final_update.T @ pt.kron(Abar, G) @ state.final_update)
        del Abar, G
        return quadratic_term + state.grad_output.T @ state.final_update + loss

    def _view_loss_gradients(self, module: pt.nn.Module) -> t.Optional[pt.Tensor]:
        return flat_tensor([param.grad for param in self.module_params[module]])

    def _update_tikhonov_reduction(self, state: ModuleState, loss: pt.Tensor):
        """Save partial information of reduction ratio to compute it next step"""
        if self._steps and self._steps % self.tikhonov_refresh_steps == 0:
            # h(theta) / (M(delta) - M(0)) = h(theta) / 0.5*gradh.T*delta
            state.reduction_denominator = 2 * loss / state.grad_output.T.dot(state.final_update)

    def _update_tikhonov_damping(self, state: ModuleState, loss: pt.Tensor):
        """Update Tikhonov damping factor (λ) using partial reduction ratio from the previous step"""
        if self._steps % self.tikhonov_refresh_steps == 1:
            # Now we can compute h(theta + delta), since the network just updated its parameters to theta + delta
            rho = loss / state.rescaling_denominator
            if rho < 1/4:
                # small "rho", increase damping factor (scaling factor is < 1)
                state.tikhonov_damping /= self.tikhonov_scaling
            elif rho > 3/4:
                # large "rho", decrease damping factor
                state.tikhonov_damping *= self.tikhonov_scaling

    # def _update_tikhonov_damping_strength(self, state: t.Dict[str, t.Any]):
    #     "ω2 = sqrt(19/20)^T2 and T2 = 10, from a starting value of λ = 150"
    #     if (self._steps + 1) % self.factored_tikhonov_refresh_steps == 0:
    #         self.factored_tikhonov_damping = -M(delta)

    def _update_natural_gradients(self, module: pt.nn.Module, state: ModuleState):
        """Compute natural gradients from the existing loss gradients

        Computing natural gradients is performed as follows:
            1. Update proposal ∆ is calculated from the approximate FIM inverse and the loss gradient wrt parameters,
            noting F already includes factored Tikhonov damping:
                `∆_t = -F^{-1}∇h(θ_t)`
            2. Final update δ rescales the update proposal and adds momentum:
                `δ_t = α.∆_t + μ.δ_{t-1}`
            3. Parameters are incremented by the final update, noting that the re-scale factor above is also interpreted
               as the learning rate:
                `θ_{t+1} = θ_t + δ_t`
        Args:
            module: Model module to process.
            state: module state containing the latest quantities needed for adjusting loss gradients.
        """
        # 1. Block-Diagional approach (this implementation)
        # approximate F^{-1} block-by-block as a diagonal of `Abar_{i-1,i-1} kron G_{i,i}`
        # 2. Block Tridiagonal approach (to be implemented)
        # approximate F^{-1} from the outer products used in 1. and the pairwise conditional covariance matrices and
        # linear coefficients.
        update_proposal = flat_tensor([
            -state.g_inv.T @ param.grad @ state.a_bar_inv
            for param in self.module_params[module]
        ])
        self._update_learning_rate_and_momentum(state, update_proposal)
        state.final_update = state.learning_rate * update_proposal + state.momentum * state.final_update

    def _update_gradients(self, module: pt.nn.Module, state: ModuleState):
        """Replace existing loss gradients with the latest natural gradients.

        Args:
            module: Model module to process.
            state: module state containing the latest quantities needed for adjusting loss gradients.
        """
        start_idx = 0
        for param in self.module_params[module]:
            if param.grad is None:
                continue
            end_idx = start_idx + param.numel()
            param.grad.set_(state.final_update[start_idx:end_idx].reshape(param.shape))
            start_idx = end_idx

    @_use_grad_for_differentiable  # noqa
    def step(self, closure: t.Optional[t.Callable[[], float]] = None) -> t.Optional[float]:
        """Performs a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        self._cuda_graph_capture_health_check()

        assert closure is not None, f"{type(self).__name__} must receive a closure providing the current loss"
        with pt.enable_grad():
            loss = closure()

        self._steps += 1
        self._update_inverses()

        for mod, state in self.module_state.items():
            self._update_tikhonov_damping(state, loss)
            self._update_natural_gradients(mod, state)
            self._update_tikhonov_reduction(state, loss)
            self._update_gradients(mod, state)

        super().step()
        if self._fallback_sgd:
            self._fallback_sgd.step()

        return loss

    def zero_grad(self, set_to_none: bool = True):
        super().zero_grad(set_to_none=set_to_none)
        if self._fallback_sgd:
            self._fallback_sgd.zero_grad(set_to_none=set_to_none)

    def _update_learning_rate_and_momentum(self, state: ModuleState, update_proposal: pt.Tensor):
        if self._steps and self._steps % self.alpha_mu_refresh_steps == 0:
            alpha, state.momentum = self._compute_rescaling_and_momentum(state, update_proposal)
            state.learning_rate = self._clip_learning_rate(state.grad_output, update_proposal, alpha)
