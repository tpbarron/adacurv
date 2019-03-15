from functools import reduce

import numpy as np
from scipy.linalg import eigvalsh_tridiagonal

import torch
from torch.optim.optimizer import Optimizer, required
from torch.nn.utils import parameters_to_vector

from adacurv.torch.optim.hvp_closures import make_fvp_fun, make_hvp_fun, make_gnvp_fun, make_fvp_fun_idx, make_gnvp_fun_idx
from adacurv.torch.utils.convert_gradients import gradients_to_vector, vector_to_gradients
from adacurv.torch.utils.cg import cg_solve
from adacurv.torch.utils.lanczos import lanczos_iteration, estimate_shrinkage

class NGD_BD(Optimizer):
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
        super(NGD_BD, self).__init__(params, defaults)

        print ("Num param groups:", len(self.param_groups))
        # if len(self.param_groups) != 1:
        #     raise ValueError("NGD-CG doesn't support per-parameter options (parameter groups)")

        # self._param_group = self.param_groups[0]
        # self._params = self._param_group['params']

        self.state = {key: {} for key in range(len(self.param_groups))}
        self._numel_cache = {key: None for key in range(len(self.param_groups))}

    def _numel(self, gi, params):
        if self._numel_cache[gi] is None:
            self._numel_cache[gi] = reduce(lambda total, p: total + p.numel(), params, 0)
        return self._numel_cache[gi]

    # def _update_worker(self, args):
    #     closure, gi, params, g, num_params, params_i, params_j, execute_update = args
    #     # print (params)
    #     group = self.param_groups[gi]
    #
    #     print ("worker on group: ", gi)
    #     state = self.state[gi]
    #     if len(state) == 0:
    #         state['step'] = 0
    #         # Set shrinkage to defaults, i.e. no shrinkage
    #         state['rho'] = 0.0
    #         state['diag_shrunk'] = 1.0
    #
    #     # print ("params: ", params[1].grad)
    #     # g = gradients_to_vector(params)
    #
    #     if 'ng_prior' not in state:
    #         state['ng_prior'] = torch.zeros_like(g)
    #
    #     curv_type = group['curv_type']
    #     # if curv_type not in self.valid_curv_types:
    #     #     raise ValueError("Invalid curv_type.")
    #
    #     # Create closure to pass to Lanczos and CG
    #     if curv_type == 'fisher':
    #         Fvp_theta_fn = make_fvp_fun(closure, params)
    #     elif curv_type == 'gauss_newton':
    #         # Pass indices instead of actual params, since these params should be the same at
    #         # the model params anyway. Then the closure should set only the subset of params
    #         # and only return the tmp_params from that subset.
    #         # This would require that the param groups are order in a specific manner?
    #         Fvp_theta_fn = make_gnvp_fun_idx(closure, params, params_i, params_j)
    #
    #     # num_params = self._numel(gi, params)
    #
    #     shrinkage_method = group['shrinkage_method']
    #     lanczos_amortization = group['lanczos_amortization']
    #     if shrinkage_method == 'lanczos' and (state['step']-1) % lanczos_amortization == 0:
    #         # print ("Computing Lanczos shrinkage at step ", state['step'])
    #         w = lanczos_iteration(Fvp_theta_fn, num_params, k=group['lanczos_iters'])
    #         rho, diag_shrunk = estimate_shrinkage(w, num_params, group['batch_size'])
    #         state['rho'] = rho
    #         state['diag_shrunk'] = diag_shrunk
    #
    #     M = None
    #     if group['cg_precondition_empirical']:
    #         # Empirical Fisher is g * g
    #         M = (g * g + group['cg_precondition_regu_coef'] * torch.ones_like(g)) ** group['cg_precondition_exp']
    #
    #     # Do CG solve with hvp fn closure
    #     extract_tridiag = group['shrinkage_method'] == 'cg'
    #     cg_result = cg_solve(Fvp_theta_fn,
    #                   g.data.clone(),
    #                   x_0=group['cg_prev_init_coef'] * state['ng_prior'],
    #                   M=M,
    #                   cg_iters=group['cg_iters'],
    #                   cg_residual_tol=group['cg_residual_tol'],
    #                   shrunk=group['shrinkage_method'] is not None,
    #                   rho=state['rho'],
    #                   Dshrunk=state['diag_shrunk'],
    #                   extract_tridiag=extract_tridiag)
    #
    #     if extract_tridiag:
    #         # print ("Computing CG shrinkage at step ", state['step'])
    #         ng, (diag_elems, off_diag_elems) = cg_result
    #         w = eigvalsh_tridiagonal(diag_elems, off_diag_elems)
    #         rho, diag_shrunk = estimate_shrinkage(w, self._numel(), group['batch_size'])
    #         state['rho'] = rho
    #         state['diag_shrunk'] = diag_shrunk
    #     else:
    #         ng = cg_result
    #
    #     state['ng_prior'] = ng.data.clone()
    #
    #     # Normalize NG
    #     lr = group['lr']
    #     alpha = torch.sqrt(torch.abs(lr / (torch.dot(g, ng) + 1e-20)))
    #
    #     # Unflatten grad
    #     vector_to_gradients(ng, params)
    #
    #     # If doing gradient ascent, reverse direction
    #     if group['ascend']:
    #         alpha *= -1
    #
    #     if execute_update:
    #         # Apply step
    #         for p in params:
    #             if p.grad is None:
    #                 continue
    #             d_p = p.grad.data
    #             p.data.add_(-alpha, d_p)
    #
    # def step(self, closure, execute_update=True):
    #     from pathos.multiprocessing import ProcessingPool as Pool
    #     p = Pool(8)
    #
    #     args = []
    #     params_i = 0
    #     params_j = 0
    #     for gi, group in enumerate(self.param_groups):
    #         params = list(group['params'])
    #         g = gradients_to_vector(params)
    #
    #         num_params = self._numel(gi, params)
    #         params_j += len(params)
    #
    #         args.append([closure, gi, params, g, num_params, params_i, params_j, execute_update])
    #         params_i = params_j
    #
    #     p.map(self._update_worker, args)


    def step(self, closure, execute_update=True):
        """Performs a single optimization step.

        Arguments:
            Fvp_fn (callable): A closure that accepts a vector of parameters and a vector of length
                equal to the number of model paramsters and returns the Fisher-vector product.
        """
        info = {}

        # If doing block diag, perform the update for each param group
        params_i = 0
        params_j = 0

        for gi, group in enumerate(self.param_groups):
            params = group['params']
            params_j += len(params)

            state = self.state[gi]
            if len(state) == 0:
                state['step'] = 0
                # Set shrinkage to defaults, i.e. no shrinkage
                state['rho'] = 0.0
                state['diag_shrunk'] = 1.0


            state['step'] += 1

            g = gradients_to_vector(params)

            if 'ng_prior' not in state:
                state['ng_prior'] = torch.zeros_like(g)

            curv_type = group['curv_type']
            if curv_type not in self.valid_curv_types:
                raise ValueError("Invalid curv_type.")

            # Create closure to pass to Lanczos and CG
            if curv_type == 'fisher':
                Fvp_theta_fn = make_fvp_fun_idx(closure, params, params_i, params_j)
            elif curv_type == 'gauss_newton':
                # Pass indices instead of actual params, since these params should be the same at
                # the model params anyway. Then the closure should set only the subset of params
                # and only return the tmp_params from that subset.
                # This would require that the param groups are order in a specific manner?
                Fvp_theta_fn = make_gnvp_fun_idx(closure, params, params_i, params_j)

            num_params = self._numel(gi, params)

            shrinkage_method = group['shrinkage_method']
            lanczos_amortization = group['lanczos_amortization']
            if shrinkage_method == 'lanczos' and (state['step']-1) % lanczos_amortization == 0:
                # print ("Computing Lanczos shrinkage at step ", state['step'])
                w = lanczos_iteration(Fvp_theta_fn, num_params, k=group['lanczos_iters'])
                rho, diag_shrunk = estimate_shrinkage(w, num_params, group['batch_size'])
                state['rho'] = rho
                state['diag_shrunk'] = diag_shrunk

            M = None
            if group['cg_precondition_empirical']:
                # Empirical Fisher is g * g
                M = (g * g + group['cg_precondition_regu_coef'] * torch.ones_like(g)) ** group['cg_precondition_exp']

            # Do CG solve with hvp fn closure
            extract_tridiag = group['shrinkage_method'] == 'cg'
            cg_result = cg_solve(Fvp_theta_fn,
                          g.data.clone(),
                          x_0=group['cg_prev_init_coef'] * state['ng_prior'],
                          M=M,
                          cg_iters=group['cg_iters'],
                          cg_residual_tol=group['cg_residual_tol'],
                          shrunk=group['shrinkage_method'] is not None,
                          rho=state['rho'],
                          Dshrunk=state['diag_shrunk'],
                          extract_tridiag=extract_tridiag)

            if extract_tridiag:
                # print ("Computing CG shrinkage at step ", state['step'])
                ng, (diag_elems, off_diag_elems) = cg_result
                w = eigvalsh_tridiagonal(diag_elems, off_diag_elems)
                rho, diag_shrunk = estimate_shrinkage(w, num_params, group['batch_size'])
                state['rho'] = rho
                state['diag_shrunk'] = diag_shrunk
            else:
                ng = cg_result

            state['ng_prior'] = ng.data.clone()

            # Normalize NG
            lr = group['lr']
            alpha = torch.sqrt(torch.abs(lr / (torch.dot(g, ng) + 1e-20)))

            # Unflatten grad
            vector_to_gradients(ng, params)

            if execute_update:
                # Apply step
                for p in params:
                    if p.grad is None:
                        continue
                    d_p = p.grad.data
                    p.data.add_(-alpha, d_p)

            params_i = params_j
            info[gi] = dict(alpha=alpha, delta=lr, natural_grad=ng)

        return info
