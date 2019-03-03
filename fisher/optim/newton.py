import numpy as np
import torch
from torch.optim.optimizer import Optimizer, required

from torch.nn.utils import parameters_to_vector
from fisher.utils.convert_gradients import gradients_to_vector, vector_to_gradients
from fisher.utils.cg import cg_solve
from fisher.utils.lanczos import lanczos_iteration, estimate_shrinkage

from functools import reduce

class Newton(Optimizer):
    r"""
    """
    def __init__(self,
                 params,
                 betas=(0.9, 0.9),
                 lr=required,
                 shrunk=False,
                 batch_size=200,
                 adaptive=False,
                 Q=None):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))

        defaults = dict(lr=lr,
                        betas=betas,
                        shrunk=shrunk,
                        batch_size=batch_size)
        if batch_size <= 0:
            raise ValueError("Batch size must be > 0")
        super(Newton, self).__init__(params, defaults)

        if len(self.param_groups) != 1:
            raise ValueError("NGD-CG doesn't support per-parameter options (parameter groups)")

        self.adaptive = adaptive
        if Q is not None:
            self.Q = torch.from_numpy(Q).float()

        self._param_group = self.param_groups[0]
        self._params = self._param_group['params']

        self.state = {}

        self._numel_cache = None

    def _numel(self):
        if self._numel_cache is None:
            self._numel_cache = reduce(lambda total, p: total + p.numel(), self._params, 0)
        return self._numel_cache

    def H(self, parameters, loss): #, damping=1e-4):
        import torch.autograd as autograd

        loss_grad = autograd.grad(loss, parameters, create_graph=True)
        cnt = 0
        for g in loss_grad:
            g_vector = g.contiguous().view(-1) if cnt == 0 else torch.cat([g_vector, g.contiguous().view(-1)])
            cnt = 1
        l = g_vector.size(0)
        hessian = torch.zeros(l, l)
        for idx in range(l):
            grad2rd = autograd.grad(g_vector[idx], parameters, create_graph=True)
            cnt = 0
            for g in grad2rd:
                g2 = g.contiguous().view(-1) if cnt == 0 else torch.cat([g2, g.contiguous().view(-1)])
                cnt = 1
            hessian[idx] = g2
        return hessian.cpu().data #.numpy()

    def step(self, loss): #, closure=None):
        """Performs a single optimization step.

        Arguments:
            Fvp_fn (callable): A closure that accepts a vector of parameters and a vector of length
                equal to the number of model paramsters and returns the Fisher-vector product.
        """
        state = self.state
        # State initialization
        if len(state) == 0:
            state['step'] = 0
            state['m'] = torch.zeros((self._numel(),))
            state['Ft'] = torch.zeros((self._numel(), self._numel()))

        state['step'] += 1

        # Get flat grad
        g = gradients_to_vector(self._params)

        # shrunk = self._param_group['shrunk']

        # Compute Fisher
        Gt = self.H(self._params, loss)

        if self.adaptive:
            m = state['m']
            Ft = state['Ft']

            beta1, beta2 = self._param_group['betas']
            bias_correction1 = 1 - beta1 ** state['step']
            bias_correction2 = 1 - beta2 ** state['step']

            m.mul_(beta1).add_(1 - beta1, g)
            g_hat = m / bias_correction1

            Ft.mul_(beta2).add_(1 - beta2, Gt)
            Ft_hat = Ft / bias_correction2

            ng = torch.pinverse(Ft_hat) @ g_hat
            H = Ft_hat
            # alpha = float(torch.sqrt(ng.dot(ng)) / (ng.view(-1, 1).t() @ Ft_hat @ ng.view(-1, 1)))
        else:
            ng = torch.pinverse(Gt) @ g
            H = Gt
            # alpha = float(torch.sqrt(ng.dot(ng)) / (ng.view(-1, 1).t() @ Gt @ ng.view(-1, 1)))

        lr = self._param_group['lr']
        alpha = torch.sqrt(torch.abs(lr / (torch.dot(g, ng) + 1e-20)))

        # alpha *= 0.1
        # Unflatten grad
        vector_to_gradients(ng, self._params)

        # Apply step
        for p in self._params:
            if p.grad is None:
                continue
            d_p = p.grad.data
            p.data.add_(-alpha, d_p)

        return dict(alpha=alpha, H=H.clone()) #, delta=lr)
