import numpy as np
import torch
from torch.optim.optimizer import Optimizer, required

from torch.nn.utils import parameters_to_vector

from fisher.optim.hvp_closures import make_fvp_fun
from fisher.utils.convert_gradients import gradients_to_vector, vector_to_gradients
from fisher.utils.cg import cg_solve
from fisher.utils.lanczos import lanczos_iteration, estimate_shrinkage

from functools import reduce

class NGD(Optimizer):
    r"""Implements "Fisher-free" natural gradient descent.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        cg_iters (int, optional): number of conjugate gradient iterations (default: 10)
        cg_residual_tol (float, optional): conjugate gradient error tolerance (default: 1e-10)
        lanczos_iters (int, optional): number of approximate eigenvalues to solve for using lanczos iteration (default: 20)
        batch_size (int, optional): training batch size, used to estimate damping when used with lanczos (default: 200)
        decay (bool, optional): whether to decay the learning rate as 1/sqrt(t) (default: True)
        ascent (bool, optional): whether to perform gradient ascent (e.g. in RL) (default: False)

    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf

    .. note::
        The implementation of SGD with Momentum/Nesterov subtly differs from
        Sutskever et. al. and implementations in some other frameworks.

        Considering the specific case of Momentum, the update can be written as

        .. math::
                  v = \rho * v + g \\
                  p = p - lr * v

        where p, g, v and :math:`\rho` denote the parameters, gradient,
        velocity, and momentum respectively.

        This is in contrast to Sutskever et. al. and
        other frameworks which employ an update of the form

       .. math::
             v = \rho * v + lr * g \\
             p = p - v

        The Nesterov version is analogously modified.
    """

    def __init__(self, params, lr=required, cg_iters=10, cg_residual_tol=1e-10, shrunk=True,
            lanczos_iters=20, batch_size=200, decay=False, ascend=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))

        defaults = dict(lr=lr,
                        cg_iters=cg_iters,
                        cg_residual_tol=cg_residual_tol,
                        shrunk=shrunk,
                        lanczos_iters=lanczos_iters,
                        batch_size=batch_size,
                        decay=decay,
                        ascend=ascend)
        if cg_iters <= 0:
            raise ValueError("CG iters must be > 0")
        if cg_residual_tol < 0:
            raise ValueError("CG residual tolerance must be >= 0")
        if shrunk and lanczos_iters <= 0:
            raise ValueError("Lanczos iters must be > 0")
        if batch_size <= 0:
            raise ValueError("Batch size must be > 0")
        super(NGD, self).__init__(params, defaults)

        if len(self.param_groups) != 1:
            raise ValueError("NGD-CG doesn't support per-parameter options (parameter groups)")

        self._param_group = self.param_groups[0]
        self._params = self._param_group['params']

        self.state = {}

        self._numel_cache = None

    def _numel(self):
        if self._numel_cache is None:
            self._numel_cache = reduce(lambda total, p: total + p.numel(), self._params, 0)
        return self._numel_cache

    # def _make_fvp_fun(self, closure, theta):
    #     """
    #     Simply create Fvp func that doesn't require theta input so that lanczos and CG can be called
    #     with generic hvp funcs.
    #     """
    #     import time
    #     s = time.time()
    #     c, _ = closure(theta)
    #     e = time.time()
    #     # print ("Closure time: ", (e-s))
    #     def f(v):
    #         hessp = Fvp(c, theta, v)
    #         return hessp.data
    #     return f
    #
    # def _make_hvp_fun(self, closure, theta):
    #     """
    #     Simply create Fvp func that doesn't require theta input so that lanczos and CG can be called
    #     with generic hvp funcs.
    #     """
    #     c, z = closure(self._params)
    #     def f(v):
    #         hessp = Hvp(c, self._params, v)
    #         return hessp.data
    #     return f
    #
    # def _make_gnvp_fun(self, closure, theta):
    #     """
    #     Simply create Fvp func that doesn't require theta input so that lanczos and CG can be called
    #     with generic hvp funcs.
    #     """
    #     c, z = closure(self._params)
    #     def f(v):
    #         hessp = GNvp_RopLop(c, z, self._params, v)
    #         return hessp.data
    #     return f

    def step(self, closure, execute_update=True): #Fvp_fn, execute_update=True, closure=None):
        """Performs a single optimization step.

        Arguments:
            Fvp_fn (callable): A closure that accepts a vector of parameters and a vector of length
                equal to the number of model paramsters and returns the Fisher-vector product.
        """
        state = self.state
        # State initialization
        if len(state) == 0:
            state['step'] = 0

        state['step'] += 1

        # Get flat grad
        g = gradients_to_vector(self._params)

        if 'ng_prior' not in state:
            state['ng_prior'] = g.data.clone()

        # Create closure to pass to Lanczos and CG
        Fvp_theta_fn = make_fvp_fun(closure, self._params)
        # Fvp_theta_fn = self._make_gnvp_fun(closure, self._params)
        # Fvp_theta_fn = self._make_hvp_fun(closure, self._params)

        # Do CG solve with hvp fn closure
        rho, diag_shrunk = 0.0, 1.0
        if self._param_group['shrunk']:
            w = lanczos_iteration(Fvp_theta_fn, self._params, k=self._param_group['lanczos_iters'])
            rho, diag_shrunk = estimate_shrinkage(w, self._numel(), self._param_group['batch_size'])

        ng = cg_solve(Fvp_theta_fn,
                      g.data.clone(),
                      cg_iters=self._param_group['cg_iters'],
                      cg_residual_tol=self._param_group['cg_residual_tol'],
                      shrunk=self._param_group['shrunk'],
                      rho=rho,
                      Dshrunk=diag_shrunk)

        state['ng_prior'] = ng.data.clone()

        # Decay LR
        if self._param_group['decay']:
            lr = self._param_group['lr'] / np.sqrt(state['step'])
        else:
            lr = self._param_group['lr']

        # Normalize NG
        alpha = torch.sqrt(torch.abs(lr / (torch.dot(g, ng) + 1e-20)))

        # Unflatten grad
        vector_to_gradients(ng, self._params)

        # If doing gradient ascent, reverse direction
        if self._param_group['ascend']:
            alpha *= -1

        if execute_update:
            # Apply step
            for p in self._params:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                p.data.add_(-alpha, d_p)

        return dict(alpha=alpha, delta=lr, natural_grad=ng)
