from functools import reduce

import numpy as np
from scipy.linalg import eigvalsh_tridiagonal

import torch
from torch.optim.optimizer import Optimizer, required
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.autograd import Variable

from adacurv.torch.optim.hvp_closures import make_fvp_fun, make_hvp_fun, make_gnvp_fun, make_fvp_fun_idx, make_gnvp_fun_idx, make_fvp_obj_fun, make_fvp_obj_fun_idx
from adacurv.torch.optim.hvp_utils import Fvp, Hvp, GNvp

from adacurv.torch.utils.convert_gradients import gradients_to_vector, vector_to_gradients
from adacurv.torch.utils.cg import cg_solve
from adacurv.torch.utils.lanczos import lanczos_iteration, estimate_shrinkage
from adacurv.torch.utils.linesearch import randomized_linesearch, randomized_linesearch_idx

class NaturalAdam_BD(Optimizer):

    def __init__(self,
                 params,
                 lr=required,
                 curv_type=required,
                 betas=(0.9, 0.99),
                 cg_iters=10,
                 cg_residual_tol=1e-10,
                 cg_prev_init_coef=0.5,
                 cg_precondition_empirical=True,
                 cg_precondition_regu_coef=0.001,
                 cg_precondition_exp=0.75,
                 shrinkage_method=None,
                 lanczos_amortization=10,
                 lanczos_iters=20,
                 batch_size=200,
                 assume_locally_linear=False):
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

        if not assume_locally_linear:
            raise ValueError("Currently only the approximate adaptive update is implemented.")

        defaults = dict(lr=lr,
                        curv_type=curv_type,
                        betas=betas,
                        cg_iters=cg_iters,
                        cg_residual_tol=cg_residual_tol,
                        cg_prev_init_coef=cg_prev_init_coef,
                        cg_precondition_empirical=cg_precondition_empirical,
                        cg_precondition_regu_coef=cg_precondition_regu_coef,
                        cg_precondition_exp=cg_precondition_exp,
                        shrinkage_method=shrinkage_method,
                        lanczos_amortization=lanczos_amortization,
                        lanczos_iters=lanczos_iters,
                        batch_size=batch_size,
                        assume_locally_linear=assume_locally_linear)
        if cg_iters <= 0:
            raise ValueError("CG iters must be > 0")
        if cg_residual_tol < 0:
            raise ValueError("CG residual tolerance must be >= 0")
        if shrinkage_method == 'lanczos' and lanczos_iters <= 0:
            raise ValueError("Lanczos iters must be > 0")
        if batch_size <= 0:
            raise ValueError("Batch size must be > 0")

        super(NaturalAdam_BD, self).__init__(params, defaults)

        print ("Num param groups:", len(self.param_groups))
        # if len(self.param_groups) != 1:
        #     raise ValueError("NGD-CG doesn't support per-parameter options (parameter groups)")

        self.state = {key: {} for key in range(len(self.param_groups))}
        self._numel_cache = {key: None for key in range(len(self.param_groups))}

    def _numel(self, gi, params):
        if self._numel_cache[gi] is None:
            self._numel_cache[gi] = reduce(lambda total, p: total + p.numel(), params, 0)
        return self._numel_cache[gi]

    def _make_combined_gnvp_fun(self, closure, theta, theta_old, group, state, params_i, params_j, bias_correction2=1.0):
        beta1, beta2 = group['betas']
        step = state['step']

        c1, z1, tmp_params1 = closure(theta_old, params_i, params_j)
        c2, z2, tmp_params2 = closure(theta, params_i, params_j)

        def f(v):
            hessp_beta1 = GNvp(c1, z1, tmp_params1, v)
            hessp_beta2 = GNvp(c2, z2, tmp_params2, v)
            if step >= 1:
                weighted_hessp = beta2 * hessp_beta1 + (1 - beta2) * hessp_beta2
            else:
                weighted_hessp = (1 - beta2) * hessp_beta2
            return weighted_hessp.data / bias_correction2
        return f

    def _make_combined_fvp_fun(self, closure, theta, theta_old, group, state, params_i, params_j, bias_correction2=1.0):
        beta1, beta2 = group['betas']
        step = state['step']

        import time
        s = time.time()
        c1, tmp_params1 = closure(theta_old, params_i, params_j)
        e = time.time()
        # print ("combined fvp closure time: ", (e-s))
        c2, tmp_params2 = closure(theta, params_i, params_j)

        def f(v):
            hessp_beta1 = Fvp(c1, tmp_params1, v)
            hessp_beta2 = Fvp(c2, tmp_params2, v)
            if step >= 1:
                weighted_hessp = beta2 * hessp_beta1 + (1 - beta2) * hessp_beta2
            else:
                weighted_hessp = (1 - beta2) * hessp_beta2
            return weighted_hessp.data / bias_correction2
        return f

    def step(self, closure, execute_update=True):
        """Performs a single optimization step.

        Arguments:
            Fvp_fn (callable): A closure that accepts a vector of parameters and a vector of length
                equal to the number of model paramsters and returns the Fisher-vector product.
        """

        # Update theta old for all blocks first, only approx update is supported
        params_i = 0
        params_j = 0

        for gi, group in enumerate(self.param_groups):
            params = group['params']
            params_j += len(params)

            num_params = self._numel(gi, params)
            # print ("num_params: ", num_params, params_i, params_j)

            state = self.state[gi]
            if len(state) == 0:
                state['step'] = 0
                # Exponential moving average of gradient values
                state['m'] = torch.zeros(num_params)
                # Maintain adaptive preconditioner if needed
                if group['cg_precondition_empirical']:
                    state['M'] = torch.zeros(num_params)
                # Set shrinkage to defaults, i.e. no shrinkage
                state['rho'] = 0.0
                state['diag_shrunk'] = 1.0
                state['lagged'] = []
                for i in range(len(params)):
                    state['lagged'].append(params[i] + torch.randn(params[i].shape) * 0.0001)

            beta1, beta2 = group['betas']

            theta = parameters_to_vector(params)
            theta_old = parameters_to_vector(state['lagged'])

            # Update theta_old beta2 portion towards theta
            theta_old = beta2 * theta_old + (1-beta2) * theta
            vector_to_parameters(theta_old, state['lagged'])
            # print (theta_old)
            # input("")

        info = {}

        # If doing block diag, perform the update for each param group
        params_i = 0
        params_j = 0

        for gi, group in enumerate(self.param_groups):
            params = group['params']
            params_j += len(params)

            num_params = self._numel(gi, params)

            # NOTE: state is initialized above
            state = self.state[gi]

            m = state['m']
            beta1, beta2 = group['betas']
            state['step'] += 1
            params_old = state['lagged'] #

            bias_correction1 = 1 - beta1 ** state['step']
            bias_correction2 = 1 - beta2 ** state['step']

            # Get flat grad
            g = gradients_to_vector(params)

            # Update moving average mean
            m.mul_(beta1).add_(1 - beta1, g)
            g_hat = m / bias_correction1

            if 'ng_prior' not in state:
                state['ng_prior'] = torch.zeros_like(g) #g_hat) #g_hat.data.clone()

            curv_type = group['curv_type']
            if curv_type not in self.valid_curv_types:
                raise ValueError("Invalid curv_type.")

            # Now that theta_old has been updated, do CG with only theta old
            if curv_type == 'fisher':
                fvp_fn_div_beta2 = make_fvp_fun_idx(closure,
                                                params_old,
                                                params_i,
                                                params_j,
                                                bias_correction2=bias_correction2)
            elif curv_type == 'gauss_newton':
                fvp_fn_div_beta2 = make_gnvp_fun(closure,
                                                params_old,
                                                bias_correction2=bias_correction2)

            shrinkage_method = group['shrinkage_method']
            lanczos_amortization = group['lanczos_amortization']
            if shrinkage_method == 'lanczos' and (state['step']-1) % lanczos_amortization == 0:
                # print ("Computing Lanczos shrinkage at step ", state['step'])
                w = lanczos_iteration(fvp_fn_div_beta2, num_params, k=group['lanczos_iters'])
                rho, diag_shrunk = estimate_shrinkage(w, num_params, group['batch_size'])
                state['rho'] = rho
                state['diag_shrunk'] = diag_shrunk

            M = None
            if group['cg_precondition_empirical']:
                # Empirical Fisher is g * g
                V = state['M']
                Mt = (g * g + group['cg_precondition_regu_coef'] * torch.ones_like(g)) ** group['cg_precondition_exp']
                Vhat = V.mul(beta2).add(1 - beta2, Mt) / bias_correction2
                V = torch.max(V, Vhat)
                M = V

            extract_tridiag = group['shrinkage_method'] == 'cg'
            cg_result = cg_solve(fvp_fn_div_beta2,
                          g_hat.data.clone(),
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
            # print ("NG: ", ng)

            state['ng_prior'] = ng.data.clone()

            # Normalize NG
            lr = group['lr']
            alpha = torch.sqrt(torch.abs(lr / (torch.dot(g_hat, ng) + 1e-20)))

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
