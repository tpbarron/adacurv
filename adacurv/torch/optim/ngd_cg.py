from functools import reduce

import numpy as np
from scipy.linalg import eigvalsh_tridiagonal

import torch
from torch.optim.optimizer import Optimizer, required
from torch.nn.utils import parameters_to_vector

from adacurv.torch.optim.hvp_closures import make_fvp_fun, make_hvp_fun, make_gnvp_fun
from adacurv.torch.utils.convert_gradients import gradients_to_vector, vector_to_gradients
from adacurv.torch.utils.cg import cg_solve
from adacurv.torch.utils.lanczos import lanczos_iteration, estimate_shrinkage

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

    def __init__(self,
                 params,
                 lr=required,
                 curv_type=required,
                 cg_iters=10,
                 cg_residual_tol=1e-10,
                 cg_prev_init_coef=0.5,
                 cg_precondition_empirical=True,
                 cg_precondition_regu_coef=0.001,
                 cg_precondition_exp=0.75,
                 shrinkage_method=None,
                 lanczos_amortization=10,
                 lanczos_iters=20,
                 batch_size=200):
        """
        Create a Truncated CG optimizer.

        params: parameters to optimizer.
            # TODO: enable multiple parameter groups and use the parameter groups to determine blocks
            for block diagonal CG.
        lr: learning rate.
        curv_type: curvature type one of fisher, gauss_newton.
        cg_iters: iterations to run CG solver (default: 10).
        cg_residual_tol: error tolerance to terminate CG (default 1e-10).
        cg_prev_init_coef: initialize the CG solver with cg_prev_init_coef * x_{t-1} (default: 0.0).
        cg_precondition_empirical: whether to precondition CG with the empirical Fisher (defaut: False)
        cg_precondition_regu_coef: regularizatin coefficient of the preconditioned empirical Fisher (default 0.001)
        cg_precondition_exp: exponent of empirical Fisher to smooth extremes (default: 0.75)
        shrinkage_method: whether to compute shrinkage and, if so, by Lanczos or CG. One of
            [None, 'lanczos', 'cg'] (default: None).
        lanczos_amortization: frequency to compute lanczos shrinkage. Only used if
            shrinkage_method='lanczos' (default: 10).
        lanczos_iters: number of iterations to run the Lanczos method. Only used if
            shrinkage_method='lanczos' (default: 20). If shrinkage_method is CG, then iterations
            defaults to cg_iters.
        batch_size: the batch size of the learning method, used for the shrinkage computation (default: 200).
            Note that this assumes constant batch size. If anyone else ever uses this and would like this to be
            more flexible please let me know.
        ascend: whether to perform gradient ascent as is RL (default: False).
            # TODO: eliminate this and just negate loss in MJRL code.
        """
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))

        self.valid_curv_types = ['fisher', 'gauss_newton']
        if curv_type is not required and curv_type not in self.valid_curv_types:
            raise ValueError("Invalid curv_type: " + str(curv_type) + ". Must be one of " + str(valid_curv_types))

        defaults = dict(lr=lr,
                        curv_type=curv_type,
                        cg_iters=cg_iters,
                        cg_residual_tol=cg_residual_tol,
                        cg_prev_init_coef=cg_prev_init_coef,
                        cg_precondition_empirical=cg_precondition_empirical,
                        cg_precondition_regu_coef=cg_precondition_regu_coef,
                        cg_precondition_exp=cg_precondition_exp,
                        shrinkage_method=shrinkage_method,
                        lanczos_amortization=lanczos_amortization,
                        lanczos_iters=lanczos_iters,
                        batch_size=batch_size)
        if cg_iters <= 0:
            raise ValueError("CG iters must be > 0")
        if cg_residual_tol < 0:
            raise ValueError("CG residual tolerance must be >= 0")
        if shrinkage_method == 'lanczos' and lanczos_iters <= 0:
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
            # Set shrinkage to defaults, i.e. no shrinkage
            state['rho'] = 0.0
            state['diag_shrunk'] = 1.0

        state['step'] += 1

        # Get flat grad
        g = gradients_to_vector(self._params)

        if 'ng_prior' not in state:
            state['ng_prior'] = torch.zeros_like(g)

        curv_type = self._param_group['curv_type']
        if curv_type not in self.valid_curv_types:
            raise ValueError("Invalid curv_type.")

        # Create closure to pass to Lanczos and CG
        if curv_type == 'fisher':
            Fvp_theta_fn = make_fvp_fun(closure, self._params)
        elif curv_type == 'gauss_newton':
            Fvp_theta_fn = make_gnvp_fun(closure, self._params)

        shrinkage_method = self._param_group['shrinkage_method']
        lanczos_amortization = self._param_group['lanczos_amortization']
        if shrinkage_method == 'lanczos' and (state['step']-1) % lanczos_amortization == 0:
            # print ("Computing Lanczos shrinkage at step ", state['step'])
            w = lanczos_iteration(Fvp_theta_fn, self._numel(), k=self._param_group['lanczos_iters'])
            rho, diag_shrunk = estimate_shrinkage(w, self._numel(), self._param_group['batch_size'])
            state['rho'] = rho
            state['diag_shrunk'] = diag_shrunk

        M = None
        if self._param_group['cg_precondition_empirical']:
            # Empirical Fisher is g * g
            M = (g * g + self._param_group['cg_precondition_regu_coef'] * torch.ones_like(g)) ** self._param_group['cg_precondition_exp']

        # Do CG solve with hvp fn closure
        extract_tridiag = self._param_group['shrinkage_method'] == 'cg'
        cg_result = cg_solve(Fvp_theta_fn,
                      g.data.clone(),
                      x_0=self._param_group['cg_prev_init_coef'] * state['ng_prior'],
                      M=M,
                      cg_iters=self._param_group['cg_iters'],
                      cg_residual_tol=self._param_group['cg_residual_tol'],
                      shrunk=self._param_group['shrinkage_method'] is not None,
                      rho=state['rho'],
                      Dshrunk=state['diag_shrunk'],
                      extract_tridiag=extract_tridiag)

        if extract_tridiag:
            # print ("Computing CG shrinkage at step ", state['step'])
            ng, (diag_elems, off_diag_elems) = cg_result
            w = eigvalsh_tridiagonal(diag_elems, off_diag_elems)
            rho, diag_shrunk = estimate_shrinkage(w, self._numel(), self._param_group['batch_size'])
            state['rho'] = rho
            state['diag_shrunk'] = diag_shrunk
        else:
            ng = cg_result

        state['ng_prior'] = ng.data.clone()

        # Normalize NG
        lr = self._param_group['lr']
        alpha = torch.sqrt(torch.abs(lr / (torch.dot(g, ng) + 1e-20)))

        # Unflatten grad
        vector_to_gradients(ng, self._params)

        if execute_update:
            # Apply step
            for p in self._params:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                p.data.add_(-alpha, d_p)

        return dict(alpha=alpha, delta=lr, natural_grad=ng)
