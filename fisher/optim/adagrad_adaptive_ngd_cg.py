import copy
from functools import reduce

import numpy as np
from scipy.linalg import eigvalsh_tridiagonal

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.optimizer import Optimizer, required
from torch.nn.utils import vector_to_parameters, parameters_to_vector

from fisher.optim.hvp_closures import make_fvp_fun, make_gnvp_fun, make_fvp_obj_fun, make_gnvp_obj_fun
from fisher.optim.hvp_utils import Fvp, Hvp, GNvp
from fisher.utils.convert_gradients import gradients_to_vector, vector_to_gradients
from fisher.utils.cg import cg_solve
from fisher.utils.lanczos import lanczos_iteration, estimate_shrinkage
from fisher.utils.linesearch import randomized_linesearch

class NaturalAdagrad(Optimizer):

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
                 batch_size=200,
                 assume_locally_linear=False):

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

        super(NaturalAdagrad, self).__init__(params, defaults)

        if len(self.param_groups) != 1:
            raise ValueError("Adaptive NGD-CG doesn't support per-parameter options (parameter groups)")

        self.state = {}

        self._numel_cache = None
        self._param_group = self.param_groups[0]
        self._params = self._param_group['params']
        self._params_old = []
        for i in range(len(self._params)):
            self._params_old.append(self._params[i] + torch.randn(self._params[i].shape) * 0.0001)

    def _numel(self):
        if self._numel_cache is None:
            self._numel_cache = reduce(lambda total, p: total + p.numel(), self._params, 0)
        return self._numel_cache

    def _make_combined_gnvp_fun(self, closure, theta, theta_old, bias_correction2=1.0):
        step = self.state['step']
        c1, z1, tmp_params1 = closure(theta_old)
        c2, z2, tmp_params2 = closure(theta)
        def f(v):
            hessp_beta1 = GNvp(c1, z1, tmp_params1, v)
            hessp_beta2 = GNvp(c2, z2, tmp_params2, v)
            if step >= 1:
                weighted_hessp = ((step - 1) * hessp_beta1 + hessp_beta2) / step
            else:
                weighted_hessp = hessp_beta2
            return weighted_hessp.data / bias_correction2
        return f

    def _make_combined_fvp_fun(self, closure, theta, theta_old, bias_correction2=1.0):
        step = self.state['step']
        c1, tmp_params1 = closure(theta_old)
        c2, tmp_params2 = closure(theta)

        def f(v):
            # TODO: execute these two HVP calls in parallel
            hessp_beta1 = Fvp(c1, tmp_params1, v)
            hessp_beta2 = Fvp(c2, tmp_params2, v)
            if step >= 1:
                weighted_hessp = ((step - 1) * hessp_beta1 + hessp_beta2) / step
            else:
                weighted_hessp = hessp_beta2
            return weighted_hessp.data / bias_correction2
        return f

    def step(self, closure, execute_update=True):
        """Performs a single optimization step.

        Arguments:
            Fvp_fn (callable): A closure that accepts a vector of length equal to the number of
                model paramsters and returns the Fisher-vector product.
        """
        state = self.state
        param_vec = parameters_to_vector(self._params)
        # State initialization
        if len(state) == 0:
            state['step'] = 0
            # Maintain adaptive preconditioner if needed
            if self._param_group['cg_precondition_empirical']:
                state['M'] = torch.zeros_like(param_vec.data)
            # Set shrinkage to defaults, i.e. no shrinkage
            state['rho'] = 0.0
            state['diag_shrunk'] = 1.0

        state['step'] += 1

        # Get flat grad
        g = gradients_to_vector(self._params)

        theta = parameters_to_vector(self._params)
        theta_old = parameters_to_vector(self._params_old)

        if 'ng_prior' not in state:
            state['ng_prior'] = torch.zeros_like(g) #g.data.clone()

        curv_type = self._param_group['curv_type']
        if curv_type not in self.valid_curv_types:
            raise ValueError("Invalid curv_type.")

        if self._param_group['assume_locally_linear'] or state['step'] == 1:
            # Update cumulative average
            theta_old = ((state['step'] - 1) * theta_old + theta) / state['step']
        else:
            # Do linesearch first to update theta_old. Then can do CG with only one HVP at each itr.
            ng = self.state['ng_prior'].clone() if state['step'] > 1 else g.data.clone()
            if curv_type == 'fisher':
                weighted_fvp_fn = self._make_combined_fvp_fun(closure, self._params, self._params_old)
                f = make_fvp_obj_fun(closure, weighted_fvp_fn, ng)
            elif curv_type == 'gauss_newton':
                weighted_fvp_fn = self._make_combined_gnvp_fun(closure, self._params, self._params_old)
                f = make_gnvp_obj_fun(closure, weighted_fvp_fn, ng)
            xmin, fmin, alpha = randomized_linesearch(f, theta_old.data, theta.data)
            theta_old = Variable(xmin.float())
        vector_to_parameters(theta_old, self._params_old)

        # Now that theta_old has been updated, do CG with only theta old
        if curv_type == 'fisher':
            fvp_fn_average = make_fvp_fun(closure, self._params_old)
        elif curv_type == 'gauss_newton':
            fvp_fn_average = make_gnvp_fun(closure, self._params_old)

        shrinkage_method = self._param_group['shrinkage_method']
        lanczos_amortization = self._param_group['lanczos_amortization']
        if shrinkage_method == 'lanczos' and (state['step']-1) % lanczos_amortization == 0:
            # print ("Computing Lanczos shrinkage at step ", state['step'])
            w = lanczos_iteration(fvp_fn_average, self._params, k=self._param_group['lanczos_iters'])
            rho, diag_shrunk = estimate_shrinkage(w, self._numel(), self._param_group['batch_size'])
            state['rho'] = rho
            state['diag_shrunk'] = diag_shrunk

        M = None
        if self._param_group['cg_precondition_empirical']:
            # Empirical Fisher is g * g
            V = state['M']
            Mt = (g * g + self._param_group['cg_precondition_regu_coef'] * torch.ones_like(g)) ** self._param_group['cg_precondition_exp']
            V = ((state['step'] - 1) * V + Mt) / state['step']
            M = V

        extract_tridiag = self._param_group['shrinkage_method'] == 'cg'
        cg_result = cg_solve(fvp_fn_average,
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

        self.state['ng_prior'] = ng.data.clone()

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
