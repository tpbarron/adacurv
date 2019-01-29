import copy
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.optimizer import Optimizer, required
from torch.nn.utils import vector_to_parameters, parameters_to_vector

from fisher.utils.convert_gradients import gradients_to_vector, vector_to_gradients
from fisher.utils.cg import cg_solve
from fisher.utils.lanczos import lanczos_iteration, estimate_shrinkage
from fisher.utils.linesearch import randomized_linesearch

from functools import reduce

class NaturalAdamExact(Optimizer):

    def __init__(self,
                 params,
                 lr=required,
                 betas=(0.9, 0.99),
                 shrunk=True,
                 lanczos_iters=20,
                 batch_size=200,
                 decay=False,
                 ascend=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))

        defaults = dict(lr=lr,
                        betas=betas,
                        shrunk=shrunk,
                        lanczos_iters=lanczos_iters,
                        batch_size=batch_size,
                        decay=decay,
                        ascend=ascend)
        if shrunk and lanczos_iters <= 0:
            raise ValueError("Lanczos iters must be > 0")
        if batch_size <= 0:
            raise ValueError("Batch size must be > 0")

        super(NaturalAdamExact, self).__init__(params, defaults)

        if len(self.param_groups) != 1:
            raise ValueError("Adaptive NGD doesn't support per-parameter options (parameter groups)")

        self.state = {}
        self._numel_cache = None

        self._param_group = self.param_groups[0]
        self._params = self._param_group['params']

    def _numel(self):
        if self._numel_cache is None:
            self._numel_cache = reduce(lambda total, p: total + p.numel(), self._params, 0)
        return self._numel_cache

    def step(self, Fvp_fn, closure=None):
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
            # Exponential moving average of gradient values
            state['m'] = torch.zeros_like(param_vec.data)
            state['Ft'] = torch.zeros((self._numel(), self._numel()))

        m = state['m']
        Ft = state['Ft']
        beta1, beta2 = self._param_group['betas']
        state['step'] += 1

        bias_correction1 = 1 - beta1 ** state['step']
        bias_correction2 = 1 - beta2 ** state['step']

        # Get flat grad
        g = gradients_to_vector(self._params)

        # Update moving average mean
        m.mul_(beta1).add_(1 - beta1, g)
        g_hat = m / bias_correction1

        # Compute Fisher
        theta = parameters_to_vector(self._params)
        G = Fvp_fn(theta.clone())
        # print (G)
        # print (Ft)
        Ft.mul_(beta2).add_(1 - beta2, G)
        Ft_hat = Ft / bias_correction2
        ng = torch.pinverse(Ft_hat) @ g_hat

        # Decay LR
        if self._param_group['decay']:
            lr = self._param_group['lr'] / np.sqrt(state['step'])
        else:
            lr = self._param_group['lr']

        # Normalize NG
        alpha = lr #torch.sqrt(torch.abs(lr / (torch.dot(g_hat, ng) + 1e-20)))

        # Unflatten grad
        vector_to_gradients(ng, self._params)

        # If doing gradient ascent, reverse direction
        if self._param_group['ascend']:
            alpha *= -1.0

        # Apply step
        for p in self._params:
            if p.grad is None:
                continue
            d_p = p.grad.data
            p.data.add_(-alpha, d_p)

        return dict(alpha=alpha, delta=lr)

    # def step(self, Fvp_fn, closure=None):
    #     """Performs a single optimization step.
    #
    #     Arguments:
    #         Fvp_fn (callable): A closure that accepts a vector of length equal to the number of
    #             model paramsters and returns the Fisher-vector product.
    #     """
    #     state = self.state
    #     param_vec = parameters_to_vector(self._params)
    #     # State initialization
    #     if len(state) == 0:
    #         state['step'] = 0
    #         # Exponential moving average of gradient values
    #         state['m'] = torch.zeros_like(param_vec.data)
    #
    #     m = state['m']
    #     beta1, beta2 = self._param_group['betas']
    #     state['step'] += 1
    #
    #     bias_correction1 = 1 - beta1 ** state['step']
    #     bias_correction2 = 1 - beta2 ** state['step']
    #
    #     # Get flat grad
    #     g = gradients_to_vector(self._params)
    #
    #     # Update moving average mean
    #     m.mul_(beta1).add_(1 - beta1, g)
    #     g_hat = m / bias_correction1
    #
    #     theta = parameters_to_vector(self._params)
    #     theta_old = parameters_to_vector(self._params_old)
    #
    #     weighted_fvp_fn_div_beta2 = self._make_combined_hvp_fun(Fvp_fn,
    #                                                             theta.clone(),
    #                                                             theta_old.clone(),
    #                                                             bias_correction2=bias_correction2)
    #     rho, diag_shrunk = 0.0, 1.0
    #     if self._param_group['shrunk']:
    #         w = lanczos_iteration(weighted_fvp_fn_div_beta2, self._params, k=self._param_group['lanczos_iters'])
    #         rho, diag_shrunk = estimate_shrinkage(w, self._numel(), self._param_group['batch_size'])
    #
    #     ng = cg_solve(weighted_fvp_fn_div_beta2,
    #                   g_hat.data.clone(),
    #                   cg_iters=self._param_group['cg_iters'],
    #                   cg_residual_tol=self._param_group['cg_residual_tol'],
    #                   shrunk=self._param_group['shrunk'],
    #                   rho=rho,
    #                   Dshrunk=diag_shrunk)
    #
    #     weighted_fvp_fn = self._make_combined_hvp_fun(Fvp_fn, theta.clone(), theta_old.clone())
    #     f = self._make_hvp_obj_fun(Fvp_fn, weighted_fvp_fn, ng.data.clone())
    #     # fprime = self._make_hvp_obj_fun_grad(Fvp_fn, weighted_fvp_fn, ng.data.clone())
    #
    #     # xmin, fmin, alpha = self.gss_linesearch(f, theta_old.data.numpy(), theta.data.numpy())
    #     # print ("gss line_search: ", fmin, alpha)
    #     xmin, fmin, alpha = self.randomized_linesearch(f, theta_old.data.numpy(), theta.data.numpy())
    #     # print ("random line_search: ", fmin, alpha)
    #     # input("")
    #
    #     theta_old = Variable(torch.from_numpy(xmin).float())
    #     vector_to_parameters(theta_old, self._params_old)
    #
    #     # Decay LR
    #     if self._param_group['decay']:
    #         lr = self._param_group['lr'] / np.sqrt(state['step'])
    #     else:
    #         lr = self._param_group['lr']
    #
    #     # Normalize NG
    #     alpha = torch.sqrt(torch.abs(lr / (torch.dot(g_hat, ng) + 1e-20)))
    #
    #     # Unflatten grad
    #     vector_to_gradients(ng, self._params)
    #
    #     # If doing gradient ascent, reverse direction
    #     if self._param_group['ascend']:
    #         alpha *= -1.0
    #
    #     # Apply step
    #     for p in self._params:
    #         if p.grad is None:
    #             continue
    #         d_p = p.grad.data
    #         p.data.add_(-alpha, d_p)
    #
    #     return dict(alpha=alpha, delta=lr)
