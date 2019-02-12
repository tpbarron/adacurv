import copy
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.optimizer import Optimizer, required
from torch.nn.utils import vector_to_parameters, parameters_to_vector

from fisher.optim.hvp_closures import make_fvp_fun, make_fvp_obj_fun #, make_combined_fvp_fun, make_fvp_obj_fun
from fisher.optim.hvp_utils import Fvp, Hvp, GNvp
from fisher.utils.convert_gradients import gradients_to_vector, vector_to_gradients
from fisher.utils.cg import cg_solve
from fisher.utils.lanczos import lanczos_iteration, estimate_shrinkage
from fisher.utils.linesearch import randomized_linesearch

from functools import reduce


class NaturalAdam(Optimizer):

    def __init__(self,
                 params,
                 lr=required,
                 betas=(0.9, 0.99),
                 cg_iters=10,
                 cg_residual_tol=1e-10,
                 shrunk=True,
                 lanczos_iters=20,
                 batch_size=200,
                 decay=False,
                 ascend=False,
                 assume_locally_linear=False):
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

        super(NaturalAdam, self).__init__(params, defaults)

        if len(self.param_groups) != 1:
            raise ValueError("Adaptive NGD-CG doesn't support per-parameter options (parameter groups)")

        self.state = {}

        self._numel_cache = None

        self._param_group = self.param_groups[0]
        self._params = self._param_group['params']
        self._params_old = []
        for i in range(len(self._params)):
            self._params_old.append(self._params[i] + torch.randn(self._params[i].shape) * 0.0001)
        self._params_tmp = []
        for i in range(len(self._params)):
            self._params_tmp.append(self._params_old[i])

    def _numel(self):
        if self._numel_cache is None:
            self._numel_cache = reduce(lambda total, p: total + p.numel(), self._params, 0)
        return self._numel_cache

    def _make_combined_fvp_fun(self, closure, theta, theta_old, bias_correction2=1.0):
        beta1, beta2 = self._param_group['betas']
        step = self.state['step']

        import time
        s = time.time()
        c1, tmp_params1 = closure(theta_old)
        c2, tmp_params2 = closure(theta)
        e = time.time()
        # print ("Closure time adam combined fvp: ", (e-s))

        def f(v):
            hessp_beta1 = Fvp(c1, tmp_params1, v)
            hessp_beta2 = Fvp(c2, tmp_params2, v)
            if step >= 1:
                weighted_hessp = beta2 * hessp_beta1 + (1 - beta2) * hessp_beta2
            else:
                weighted_hessp = (1 - beta2) * hessp_beta2
            return weighted_hessp.data / bias_correction2
        return f

    # def _make_fvp_obj_fun(self, closure, weighted_fvp_fn, ng):
    #     v2 = weighted_fvp_fn(ng)
    #     def f(p):
    #         pvar = Variable(torch.from_numpy(p).float(), requires_grad=False)
    #         # vector_to_parameters(pvar, self._params_tmp)
    #         import time
    #         s = time.time()
    #         c, tmp_params = closure(pvar)
    #         e = time.time()
    #         # print ("Closure time: ", (e-s))
    #         v1 = Fvp(c, tmp_params, ng)
    #         # v2 = weighted_fvp_fn(ng)
    #         loss = F.mse_loss(v1, v2)
    #         return float(loss.data)
    #     return f

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

        if self._param_group['assume_locally_linear']:
            # print ("NO LINE SEARCH")
            import time
            s = time.time()
            # Update theta_old beta2 portion towards theta
            theta_old = beta2 * theta_old + (1-beta2) * theta
            e = time.time()
            # print ("Approx time: ", (e-s))
        else:
            # Do linesearch first to update theta_old. Then can do CG with only one HVP at each itr.
            # weighted_fvp_fn = self._make_combined_fvp_fun(Fvp_fn, theta.clone(), theta_old.clone())
            weighted_fvp_fn = self._make_combined_fvp_fun(closure, self._params, self._params_old) #theta, theta_old)
            # f = self._make_fvp_obj_fun(Fvp_fn, weighted_fvp_fn, self.state['ng_prior'].clone())
            f = make_fvp_obj_fun(closure, weighted_fvp_fn, self.state['ng_prior'].clone())
            # import time
            # s = time.time()
            xmin, fmin, alpha = randomized_linesearch(f, theta_old.data, theta.data)
            # e = time.time()
            # print ("LS time: ", (e-s))
            # theta_old = Variable(torch.from_numpy(xmin).float())
            theta_old = Variable(xmin.float())

        vector_to_parameters(theta_old, self._params_old)

        # import time
        # s = time.time()
        # Now that theta_old has been updated, do CG with only theta old
        fvp_fn_div_beta2 = make_fvp_fun(closure,
                                        self._params_old, #theta_old.clone(),
                                        bias_correction2=bias_correction2)

        # e = time.time()
        # print ("fvp fn div beta2 time: ", (e-s))

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
