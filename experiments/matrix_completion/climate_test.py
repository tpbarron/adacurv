
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import adacurv.torch.optim as fisher_optim
from torch.nn.utils import vector_to_parameters, parameters_to_vector
from torch.optim.lr_scheduler import LambdaLR

class Factorization(nn.Module):

    def __init__(self, m, n, r):
        super(Factorization, self).__init__()
        self.A = nn.Parameter(torch.rand(m, r) * 0.1)
        self.B = nn.Parameter(torch.rand(n, r) * 0.1)

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
    # print ("W: ", W.shape, W[ids].shape, P.shape)
    e = torch.norm(W[ids] * (M[ids] - P), 'fro') ** 2.0
    regu = lmda1 * torch.norm(A, 'fro') ** 2.0 + lmda2 * torch.norm(B, 'fro') ** 2.0
    # regu = lmda1 * torch.norm(A, 'fro') + lmda2 * torch.norm(B, 'fro')
    return e + regu

def build_mat_completion_loss_closure_combined(model, W, M, ids):
    def func(params):
        # import time
        # t1 = time.time()
        old_params = parameters_to_vector(model.parameters())
        if isinstance(params, Variable):
            vector_to_parameters(params, model.parameters())
        else:
            vector_to_parameters(parameters_to_vector(params), model.parameters())

        z = model(ids)
        f = mat_completion_loss(W, M, z, model.A, model.B, ids)

        tmp_params = list(model.parameters())
        vector_to_parameters(old_params, model.parameters())
        # t2 = time.time()
        # print ("Loss closure time: ", (t2 - t1))
        return f, z, tmp_params
    return func


def randomize_windices(W):
    Windx, Windy = W
    perm = np.random.permutation(len(Windx))
    Windx = Windx[perm]
    Windy = Windy[perm]
    return Windx, Windy


def run(rcp='rcp45', iters=500, batch=2500, use_gn=True, seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)

    # M in (m x n) of rank r
    # A is (m x r)
    # B is (n x r)
    # So A @ B.T is (m x n)

    M = np.load("data/climate_data/matrices/M_1900_2101_"+rcp+".npy")
    W = np.load("data/climate_data/matrices/W_1900_2101_"+rcp+".npy")

    bs = batch
    n_samples = np.count_nonzero(W)
    print ("num sampes: ", n_samples)
    Wind = np.nonzero(W)
    Wind = randomize_windices(Wind)
    n_batches = int(np.ceil(n_samples / bs))
    print ("n_batches: ", n_batches)

    r = 5
    m = M.shape[0]
    n = M.shape[1]
    M = torch.from_numpy(M).float()
    W = torch.from_numpy(W).float()

    fac = Factorization(m, n, r)

    print ("r, m, n: ", r, m, n)
    print ("A, B: ", fac.A.shape, fac.B.shape)

    if use_gn:
        optA = fisher_optim.NaturalAdam([fac.A, fac.B],
                                         lr=0.01,
                                         curv_type='gauss_newton',
                                         cg_prev_init_coef=0.0,
                                         cg_precondition_empirical=False,
                                         shrinkage_method=None,
                                         batch_size=bs,
                                         betas=(0.1, 0.1),
                                         assume_locally_linear=True)
    else:
        optA = optim.Adam([fac.A, fac.B], lr=0.01)

    P = fac(ids=Wind)
    print ("P init: ", P.shape)
    init_error = mat_completion_loss(W, M, P, fac.A, fac.B, Wind)
    print ("Init error: ", init_error / P.shape[0])

    # input("Start training?")
    best_error = float(init_error)

    from torch.optim.lr_scheduler import ReduceLROnPlateau
    scheduler = ReduceLROnPlateau(optA, 'min')

    for i in range(iters):
        Wind = randomize_windices(Wind)
        for j in range(n_batches):

            optA.zero_grad()

            ind1 = j*bs
            ind2 = (j+1)*bs

            Windx, Windy = Wind
            batch_idx_idy = Windx[ind1:ind2], Windy[ind1:ind2]

            P = fac(ids=batch_idx_idy)

            error = mat_completion_loss(W, M, P, fac.A, fac.B, batch_idx_idy)
            error.backward()

            if use_gn:
                # import time
                # t1 = time.time()
                loss_closure = build_mat_completion_loss_closure_combined(fac, W, M, batch_idx_idy)
                # t2 = time.time()
                # print ("Building loss closure time: ", (t2 - t1))
                optA.step(loss_closure)
            else:
                optA.step()

            if j % 10 == 0:
                print ("Iter: ", i, ", batch: ", j, float(error) / P.shape[0])

        P = fac(Wind)
        error = float(mat_completion_loss(W, M, P, fac.A, fac.B, Wind))
        print ("Iter: ", i, error / P.shape[0])
        scheduler.step(error)

        if error < best_error:
            P2 = fac()
            gn_str = 'gn' if use_gn else 'adam'
            np.save('models/P_'+rcp+'_rank'+str(r)+'_'+gn_str+'.npy', P2.data.numpy())
            best_error = error



if __name__ == "__main__":
    run(rcp='rcp45')
    run(rcp='rcp85')
