import copy
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.optimizer import Optimizer, required
from torch.nn.utils import vector_to_parameters, parameters_to_vector

from fisher.optim.hvp_closures import make_fvp_fun, make_fvp_obj_fun
from fisher.optim.hvp_utils import Fvp, Hvp, GNvp

from fisher.utils.convert_gradients import gradients_to_vector, vector_to_gradients
from fisher.utils.cg import cg_solve
from fisher.utils.lanczos import lanczos_iteration, estimate_shrinkage
from fisher.utils.linesearch import randomized_linesearch

from functools import reduce

class NaturalAdagrad(Optimizer):

    def __init__(self, params, lr=required, cg_iters=10, cg_residual_tol=1e-10,
            shrunk=True, lanczos_iters=20, batch_size=200, decay=False, ascend=False, assume_locally_linear=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))

        defaults = dict(lr=lr,
                        cg_iters=cg_iters,
                        cg_residual_tol=cg_residual_tol,
                        shrunk=shrunk,
                        lanczos_iters=lanczos_iters,
                        batch_size=batch_size,
                        decay=decay,
                        ascend=ascend,
                        assume_locally_linear=assume_locally_linear)
        if cg_iters <= 0:
            raise ValueError("CG iters must be > 0")
        if cg_residual_tol < 0:
            raise ValueError("CG residual tolerance must be >= 0")
        if shrunk and lanczos_iters <= 0:
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

    def step(self, closure, execute_update=True): #Fvp_fn, execute_update=True, closure=None):
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

        state['step'] += 1

        # Get flat grad
        g = gradients_to_vector(self._params)

        theta = parameters_to_vector(self._params)
        theta_old = parameters_to_vector(self._params_old)

        if 'ng_prior' not in state:
            state['ng_prior'] = g.data.clone()

        if self._param_group['assume_locally_linear'] or state['step'] == 1:
            # Update cumulative average
            theta_old = ((state['step'] - 1) * theta_old + theta) / state['step']
        else:
            # Do linesearch first to update theta_old. Then can do CG with only one HVP at each itr.
            weighted_fvp_fn = self._make_combined_fvp_fun(closure, self._params, self._params_old)
            f = make_fvp_obj_fun(closure, weighted_fvp_fn, self.state['ng_prior'].clone())
            xmin, fmin, alpha = randomized_linesearch(f, theta_old.data, theta.data)
            theta_old = Variable(xmin.float())
        vector_to_parameters(theta_old, self._params_old)

        # Now that theta_old has been updated, do CG with only theta old
        fvp_fn_average = make_fvp_fun(closure, self._params_old)

        rho, diag_shrunk = 0.0, 1.0
        if self._param_group['shrunk']:
            w = lanczos_iteration(fvp_fn_average, self._params, k=self._param_group['lanczos_iters'])
            rho, diag_shrunk = estimate_shrinkage(w, self._numel(), self._param_group['batch_size'])

        ng = cg_solve(fvp_fn_average,
                      g.data.clone(),
                      cg_iters=self._param_group['cg_iters'],
                      cg_residual_tol=self._param_group['cg_residual_tol'],
                      shrunk=self._param_group['shrunk'],
                      rho=rho,
                      Dshrunk=diag_shrunk)

        self.state['ng_prior'] = ng.data.clone()

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
            alpha *= -1.0

        if execute_update:
            # Apply step
            for p in self._params:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                p.data.add_(-alpha, d_p)

        return dict(alpha=alpha, delta=lr, natural_grad=ng)
