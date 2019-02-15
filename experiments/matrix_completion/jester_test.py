
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import fisher.optim as fisher_optim
from torch.nn.utils import vector_to_parameters, parameters_to_vector

class Factorization(nn.Module):

    def __init__(self, m, n, r):
        super(Factorization, self).__init__()
        self.A = nn.Parameter(torch.randn(m, r) * 0.01)
        self.B = nn.Parameter(torch.randn(n, r) * 0.01)

    def forward(self):
        P = self.A @ self.B.t()
        return P

def mat_completion_loss(W, M, P, A, B, lmda1=0.01, lmda2=0.01):
    e = torch.norm(W * (M - P), 'fro') ** 2.0
    regu = lmda1 * torch.norm(A, 'fro') ** 2.0 + lmda2 * torch.norm(B, 'fro') ** 2.0
    return e + regu

def build_mat_completion_loss_closure(model, W, M):
    def func(params):
        # print ("MAT Comp loss closure")
        old_params = parameters_to_vector(model.parameters())
        if isinstance(params, Variable):
            vector_to_parameters(params, model.parameters())
        else:
            vector_to_parameters(parameters_to_vector(params), model.parameters())
        z = model()
        f = mat_completion_loss(W, M, z, model.A, model.B)
        tmp_params = list(model.parameters())
        vector_to_parameters(old_params, model.parameters())
        return f, z, tmp_params

    return func


if __name__ == "__main__":
    # M in (m x n) of rank r
    # A is (m x r)
    # B is (n x r)
    # So A @ B.T is (m x n)

    from scipy.io import loadmat
    jester = loadmat('lrmf-datasets/Jester_1/jester_1.mat')
    print (jester['M'].shape)
    print (jester['W'].shape)

    # 100 jokes from 24983 users
    M = np.nan_to_num(jester['M'] / 10.0, 0.0)
    W = jester['W']
    # print (M[0:10,0:10])
    # print (W[0:10,0:10])

    r = 20
    m = M.shape[0]
    n = M.shape[1]
    M = torch.from_numpy(M).float()
    W = torch.from_numpy(W).float()

    fac = Factorization(m, n, r)
    # opt = optim.Adam(fac.parameters(), lr=0.01)
    opt = fisher_optim.NGD(fac.parameters(),
                                 lr=0.1,
                                 shrunk=False,
                                 lanczos_iters=20,
                                 batch_size=500)
    # opt = fisher_optim.NaturalAdam(fac.parameters(),
    #                                  lr=0.1,
    #                                  shrunk=False,
    #                                  lanczos_iters=-1,
    #                                  batch_size=10)

    P = fac()
    init_error = mat_completion_loss(W, M, P, fac.A, fac.B)
    print ("Init error: ", init_error)


    for i in range(20000):
        opt.zero_grad()
        P = fac()
        error = mat_completion_loss(W, M, P, fac.A, fac.B)
        error.backward()

        loss_closure = build_mat_completion_loss_closure(fac, W, M)
        opt.step(loss_closure)
        # opt.step()
        if i % 2 == 0:
            print ("Iter: ", i, float(error))
