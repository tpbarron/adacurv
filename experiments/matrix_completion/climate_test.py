
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

    def forward(self, ids=None):
        if ids is not None:
            Asub = self.A[ids[0],:]
            Bsub = self.B[ids[1],:]
            # print (ids[0], len(np.unique(ids[0])))
            # print ("shapes: ", Asub.shape, Bsub.shape, self.A.shape, self.B.shape)
            P = torch.sum(Asub * Bsub, dim=1)
        else:
            P = self.A @ self.B.t()
        return P

def mat_completion_loss(W, M, P, A, B, ids, lmda1=0.001, lmda2=0.001):
    # e = torch.norm(W[ids] * (M[ids] - P[ids]), 'fro') #** 2.0
    e = torch.norm(W[ids] * (M[ids] - P), 'fro') ** 2.0
    regu = lmda1 * torch.norm(A, 'fro') ** 2.0 + lmda2 * torch.norm(B, 'fro') ** 2.0
    # regu = lmda1 * torch.norm(A, 'fro') + lmda2 * torch.norm(B, 'fro')
    return e + regu

def build_mat_completion_loss_closure_combined(model, W, M, ids):
    def func(params):
        old_params = parameters_to_vector(model.parameters())
        if isinstance(params, Variable):
            vector_to_parameters(params, model.parameters())
        else:
            vector_to_parameters(parameters_to_vector(params), model.parameters())

        z = model(ids)
        f = mat_completion_loss(W, M, z, model.A, model.B, ids)

        tmp_params = list(model.parameters())
        vector_to_parameters(old_params, model.parameters())
        return f, z, tmp_params
    return func


def build_mat_completion_loss_closure(model, W, M, ids, mat='A'):
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

        z = model(ids)
        f = mat_completion_loss(W, M, z, model.A, model.B, ids)
        if mat == 'A':
            tmp_params = [model.A]
        elif mat == 'B':
            tmp_params = [model.B]

        vector_to_parameters(old_params, model.parameters())
        return f, z, tmp_params

    return func

def randomize_windices(W):
    Windx, Windy = W
    perm = np.random.permutation(len(Windx))
    Windx = Windx[perm]
    Windy = Windy[perm]
    return Windx, Windy

if __name__ == "__main__":
    # M in (m x n) of rank r
    # A is (m x r)
    # B is (n x r)
    # So A @ B.T is (m x n)

    M = np.load("data/climate_data/matrices/M_1900_2100_rcp26.npy")
    W = np.load("data/climate_data/matrices/W_1900_2100_rcp26.npy")


    bs = 1000
    n_samples = np.count_nonzero(W)
    print ("num sampes: ", n_samples)
    Wind = np.nonzero(W)
    Wind = randomize_windices(Wind)
    n_batches = int(np.ceil(n_samples / bs))
    print ("n_batches: ", n_batches)

    r = 10
    m = M.shape[0]
    n = M.shape[1]
    M = torch.from_numpy(M).float()
    W = torch.from_numpy(W).float()

    fac = Factorization(m, n, r)

    print ("r, m, n: ", r, m, n)
    print ("A, B: ", fac.A.shape, fac.B.shape)

    gn = True
    if gn:
        # optA = fisher_optim.NGD([fac.A, fac.B],
        #                        lr=0.01,
        #                        curv_type='gauss_newton',
        #                        shrinkage_method=None, #'cg', #'lanzcos',
        #                        lanczos_iters=0,
        #                        batch_size=bs)
        # optB = fisher_optim.NGD([fac.B],
        #                        lr=0.001,
        #                        curv_type='gauss_newton',
        #                        shrinkage_method=None, #'cg', #'lanzcos',
        #                        lanczos_iters=0,
        #                        batch_size=bs)
        optA = fisher_optim.NaturalAdam([fac.A, fac.B],
                                         lr=0.01,
                                         curv_type='gauss_newton',
                                         shrinkage_method=None,
                                         batch_size=bs,
                                         betas=(0.1, 0.1),
                                         assume_locally_linear=True)
        # optB = fisher_optim.NaturalAdam([fac.B],
        #                                  lr=0.001,
        #                                  curv_type='gauss_newton',
        #                                  shrinkage_method=None,
        #                                  batch_size=bs,
        #                                  betas=(0.9, 0.9),
        #                                  assume_locally_linear=True)
    else:
        optA = optim.Adam([fac.A, fac.B], lr=0.01)
        # optB = optim.Adam([fac.B], lr=0.001)

    P = fac(ids=Wind)
    print ("P init: ", P.shape)
    init_error = mat_completion_loss(W, M, P, fac.A, fac.B, Wind)
    print ("Init error: ", init_error / P.shape[0])

    # lambda_lr = lambda epoch: 0.9 #1.0 / np.sqrt(epoch+1)
    # scheduler = LambdaLR(opt, lr_lambda=[lambda_lr])

    input("Start training?")

    for i in range(150):
        Wind = randomize_windices(Wind)
        for j in range(n_batches):

            optA.zero_grad()
            # optB.zero_grad()

            ind1 = j*bs
            ind2 = (j+1)*bs

            Windx, Windy = Wind
            batch_idx_idy = Windx[ind1:ind2], Windy[ind1:ind2]

            P = fac(ids=batch_idx_idy)

            error = mat_completion_loss(W, M, P, fac.A, fac.B, batch_idx_idy)
            error.backward()

            if gn:
                loss_closure = build_mat_completion_loss_closure_combined(fac, W, M, batch_idx_idy)
                # loss_closure = build_mat_completion_loss_closure(fac, W, M, batch_idx_idy, mat='A')
                optA.step(loss_closure)
                # loss_closure = build_mat_completion_loss_closure(fac, W, M, batch_idx_idy, mat='B')
                # optB.step(loss_closure)
            else:
                optA.step()
                # optB.step()

            if j % 10 == 0:
                print ("Iter: ", i, ", batch: ", j, float(error) / P.shape[0])

        P = fac(Wind)
        error = mat_completion_loss(W, M, P, fac.A, fac.B, Wind)
        print ("Iter: ", i, float(error) / P.shape[0])


        # # scheduler.step()
