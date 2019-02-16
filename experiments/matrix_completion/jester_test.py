
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import fisher.optim as fisher_optim
from torch.nn.utils import vector_to_parameters, parameters_to_vector
from torch.optim.lr_scheduler import LambdaLR

class Factorization(nn.Module):

    def __init__(self, m, n, r):
        super(Factorization, self).__init__()
        self.A = nn.Parameter(torch.randn(m, r) * 0.01)
        self.B = nn.Parameter(torch.randn(n, r) * 0.01)

    def forward(self):
        P = self.A @ self.B.t()
        return P

def mat_completion_loss(W, M, P, A, B, lmda1=0.001, lmda2=0.001):
    e = torch.norm(W * (M - P), 'fro') #** 2.0
    # regu = lmda1 * torch.norm(A, 'fro') ** 2.0 + lmda2 * torch.norm(B, 'fro') ** 2.0
    regu = lmda1 * torch.norm(A, 'fro') + lmda2 * torch.norm(B, 'fro')
    return e + regu

def build_mat_completion_loss_closure(model, W, M, mat='A'):
    def func(params):
        old_params = parameters_to_vector(model.parameters())

        # print ("old: ", len(old_params), len(list(model.parameters())))
        # print (type(params), type(params[0]))
        if isinstance(params[0], torch.nn.Parameter):
            # print ("1")
            if mat == 'A':
                vector_to_parameters(params[0].view(-1, 1), model.A) #parameters())
            elif mat == 'B':
                vector_to_parameters(params[0].view(-1, 1), model.B) #parameters())
        else:
            # print ("2")
            if mat == 'A':
                vector_to_parameters(parameters_to_vector(params[0]), model.A) #parameters())
            elif mat == 'B':
                vector_to_parameters(parameters_to_vector(params[0]), model.B) #parameters())

        # if isinstance(params[0], torch.nn.Parameter):
        #     print ("params is nn.param")
        #     p = params[0]
        # elif isinstance(params[0], torch.Tensor):
        #     print ("params is tensor")
        #     if mat == 'A':
        #         p = params[0].view_as(model.A) #torch.nn.Parameter(params[0].view_as(model.A))
        #         # p = torch.nn.Parameter(params[0].view_as(model.A))
        #     elif mat == 'B':
        #         p = params[0].view_as(model.B) #torch.nn.Parameter(params[0].view_as(model.A))
        #         # p = torch.nn.Parameter(params[0].view_as(model.B))

        # if mat == 'A':
        #     model.A = p # torch.nn.Parameter(p.view_as(model.A)) #params[0]
        # elif mat == 'B':
        #     model.B = p # torch.nn.Parameter(p.view_as(model.B)) #params[0]

        z = model()
        f = mat_completion_loss(W, M, z, model.A, model.B)
        if mat == 'A':
            tmp_params = [model.A]
        elif mat == 'B':
            tmp_params = [model.B]

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

    r = 5
    m = M.shape[0]
    n = M.shape[1]
    M = torch.from_numpy(M).float()
    W = torch.from_numpy(W).float()

    fac = Factorization(m, n, r)

    print ("r, m, n: ", r, m, n)
    print ("A, B: ", fac.A.shape, fac.B.shape)

    gn = True
    if gn:
        # optA = fisher_optim.NGD([fac.A],
        #                        lr=0.1,
        #                        curv_type='gauss_newton',
        #                        shrinkage_method=None, #'lanzcos',
        #                        lanczos_iters=20,
        #                        batch_size=100)
        # optB = fisher_optim.NGD([fac.B],
        #                        lr=0.1,
        #                        curv_type='gauss_newton',
        #                        shrinkage_method=None, #'lanzcos',
        #                        lanczos_iters=20,
        #                        batch_size=100)
        optA = fisher_optim.NaturalAdam([fac.A],
                                         lr=0.1,
                                         curv_type='gauss_newton',
                                         shrinkage_method=None,
                                         batch_size=100,
                                         betas=(0.0, 0.0))
        optB = fisher_optim.NaturalAdam([fac.B],
                                         lr=0.1,
                                         curv_type='gauss_newton',
                                         shrinkage_method=None,
                                         batch_size=24983,
                                         betas=(0.0, 0.0))


    else:
        # opt = optim.Adam(fac.parameters(), lr=0.1)
        optA = optim.Adam([fac.A], lr=0.1)
        optB = optim.Adam([fac.B], lr=0.1)

    P = fac()
    init_error = mat_completion_loss(W, M, P, fac.A, fac.B)
    print ("Init error: ", init_error)

    # lambda_lr = lambda epoch: 0.9 #1.0 / np.sqrt(epoch+1)
    # scheduler = LambdaLR(opt, lr_lambda=[lambda_lr])

    for i in range(200):
        optA.zero_grad()
        optB.zero_grad()

        P = fac()
        error = mat_completion_loss(W, M, P, fac.A, fac.B)
        error.backward()

        if gn:
            # print ("Pre A")
            loss_closure = build_mat_completion_loss_closure(fac, W, M, mat='A')
            optA.step(loss_closure)
            # print ("Post A")

            # print ("Pre B")
            loss_closure = build_mat_completion_loss_closure(fac, W, M, mat='B')
            optB.step(loss_closure)
            # print ("Post B")
            # input("")
        else:
            optA.step()
            optB.step()
            # opt.step()
        if i % 1 == 0:
            print ("Iter: ", i, float(error))

        # scheduler.step()
