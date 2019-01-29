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

class NaturalAmsgrad(Optimizer):

    def __init__(self, params, lr=required, betas=(0.9, 0.99), cg_iters=10, cg_residual_tol=1e-10,
            shrunk=True, lanczos_iters=20, batch_size=200, decay=False, ascend=False, assume_locally_linear=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))

        defaults = dict(lr=lr,
                        betas=betas,
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

        super(NaturalAmsgrad, self).__init__(params, defaults)

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

    def _make_hvp_fun(self, Fvp_fn, theta, bias_correction2=1.0):
        def f(v):
            hessp = Fvp_fn(theta, v)
            return hessp.data / bias_correction2
        return f

    def _make_combined_hvp_fun(self, Fvp_fn, theta, theta_old, bias_correction2=1.0):
        beta1, beta2 = self._param_group['betas']
        step = self.state['step']
        def f(v):
            # TODO: execute these two HVP calls in parallel
            hessp_beta1 = Fvp_fn(theta_old, v)
            hessp_beta2 = Fvp_fn(theta, v)
            if step >= 1:
                weighted_hessp = beta2 * hessp_beta1 + (1 - beta2) * hessp_beta2
            else:
                weighted_hessp = (1 - beta2) * hessp_beta2
            return weighted_hessp.data / bias_correction2
        return f

    def _make_hvp_obj_fun(self, Fvp_fn, weighted_fvp_fn, ng):
        beta1, beta2 = self._param_group['betas']
        def f(p):
            pvar = Variable(torch.from_numpy(p).float(), requires_grad=False)
            # TODO: execute these two calls in parallel
            v1 = Fvp_fn(pvar, ng)
            v2 = weighted_fvp_fn(ng)
            loss = F.mse_loss(v1, v2)
            return float(loss.data)
        return f

    def _make_hvp_obj_fun_grad(self, Fvp_fn, weighted_fvp_fn, ng):
        beta1, beta2 = self._param_group['betas']
        def f(p):
            pvar = Variable(torch.from_numpy(p).float(), requires_grad=True)
            # Need temp model for grads to pass
            # TODO: execute these two calls in parallel
            v1, model = Fvp_fn(pvar, ng, return_model=True)
            v2 = weighted_fvp_fn(ng)
            loss = F.mse_loss(v1, v2)
            grad = torch.autograd.grad(loss, model.parameters())
            grad_vec = torch.cat([g.contiguous().view(-1) for g in grad]).numpy().astype(np.float64)
            return grad_vec
        return f

    def step(self, Fvp_fn, execute_update=True, closure=None):
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

        m = state['m']
        beta1, beta2 = self._param_group['betas']
        state['step'] += 1

        bias_correction1 = 1 - beta1 ** state['step']
        bias_correction2 = 1 - beta2 ** state['step']

        # Get flat grad
        g = gradients_to_vector(self._params)

        # Update moving average mean
        m.mul_(beta1).add_(1 - beta1, g)
        g_hat = m / bias_correction1

        theta = parameters_to_vector(self._params)
        theta_old = parameters_to_vector(self._params_old)

        if 'ng_prior' not in state:
            state['ng_prior'] = g_hat.data.clone()
        if 'max_fisher_spectral_norm' not in state:
            state['max_fisher_spectral_norm'] = 0.0

        weighted_fvp_fn_div_beta2 = self._make_combined_hvp_fun(Fvp_fn,
                                                                theta.clone(),
                                                                theta_old.clone(),
                                                                bias_correction2=bias_correction2)

        fisher_norm = lanczos_iteration(weighted_fvp_fn_div_beta2, self._params, k=1)[0]
        is_max_norm = fisher_norm > state['max_fisher_spectral_norm'] or state['step'] == 1
        if is_max_norm:
            state['max_fisher_spectral_norm'] = fisher_norm

        if is_max_norm:
            if self._param_group['assume_locally_linear']:
                # Update theta_old beta2 portion towards theta
                theta_old = beta2 * theta_old + (1-beta2) * theta
            else:
                # Do linesearch first to update theta_old. Then can do CG with only one HVP at each itr.
                weighted_fvp_fn = self._make_combined_hvp_fun(Fvp_fn, theta.clone(), theta_old.clone())
                f = self._make_hvp_obj_fun(Fvp_fn, weighted_fvp_fn, self.state['ng_prior'].clone())
                xmin, fmin, alpha = randomized_linesearch(f, theta_old.data.numpy(), theta.data.numpy())
                theta_old = Variable(torch.from_numpy(xmin).float())
            vector_to_parameters(theta_old, self._params_old)

        # Now that theta_old has been updated, do CG with only theta old
        # If not max norm, then this will remain the old params.
        fvp_fn_div_beta2 = self._make_hvp_fun(Fvp_fn,
                                              theta_old.clone(),
                                              bias_correction2=bias_correction2)

        rho, diag_shrunk = 0.0, 1.0
        if self._param_group['shrunk']:
            w = lanczos_iteration(fvp_fn_div_beta2, self._params, k=self._param_group['lanczos_iters'])
            rho, diag_shrunk = estimate_shrinkage(w, self._numel(), self._param_group['batch_size'])

        ng = cg_solve(fvp_fn_div_beta2,
                      g_hat.data.clone(),
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
        alpha = torch.sqrt(torch.abs(lr / (torch.dot(g_hat, ng) + 1e-20)))
        # alpha = torch.sqrt(torch.abs(self._param_group['lr'] / (torch.dot(g_hat, ng) + 1e-20)))

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
    #     self._param_group['lr'] *= self._param_group['decay']
    #     lr = self._param_group['lr'] / np.sqrt(state['step'])
    #
    #     # Normalize NG
    #     alpha = torch.sqrt(torch.abs(lr / (torch.dot(g_hat, ng) + 1e-20)))
    #     # alpha = torch.sqrt(torch.abs(self._param_group['lr'] / (torch.dot(g_hat, ng) + 1e-20)))
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
    #     return None
