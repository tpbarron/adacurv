import numpy as np

import torch
from torch.nn.utils import parameters_to_vector

from scipy.linalg import eigvalsh_tridiagonal
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import eigsh

def lanczos_iteration_scipy(Fvp_fn, params, k=20):
    """
    Fvp_fn must have parameters closure
    That is
    """
    theta = parameters_to_vector(params)
    n_params = len(theta)
    def mv(v):
        v = torch.from_numpy(v).float()
        hvp_p = Fvp_fn(v)
        return hvp_p.data.numpy()
    H = LinearOperator((n_params, n_params), matvec=mv)
    try:
        w = eigsh(H, k=k, which='LM', return_eigenvectors=False)
    except ArpackNoConvergence as arpack:
        w = arpack.eigenvalues
    return w

def lanczos_iteration(Fvp_fn, dim, k=20):
    v = torch.FloatTensor(dim).uniform_()
    v /= torch.norm(v, 2)

    diag = []
    diag_adj = []

    w = Fvp_fn(v)
    alpha = w.dot(v)
    w -= alpha * v
    diag.append(alpha)

    for i in range(k-1):
        beta = torch.norm(w, 2)
        if beta == 0:
            break
        v_prev = v.clone()
        v = w / beta
        w = Fvp_fn(v)
        alpha = w.dot(v)
        diag.append(alpha)
        diag_adj.append(beta)
        w = w - alpha * v - beta * v_prev

    diag, diag_adj = np.array(diag), np.array(diag_adj)
    w = eigvalsh_tridiagonal(np.array(diag), np.array(diag_adj))
    return w

def estimate_shrinkage_buggy(eigvals, p, batch_size):
    # p = len(eigvals)
    # Tr(s) = Sum(\lambda_i)
    trS = np.sum(eigvals)
    # print ("trS: ", trS)

    # Tr^2(S) - computed knowing that Tr(s) = Sum(eigvals(S))
    tr2S = trS**2.0
    # print ("tr2S: ", tr2S)

    # TODO: optimize this.
    coef = 0.0
    for j in range(len(eigvals)):
        for i in range(j):
            coef += eigvals[j] * eigvals[i]
    trS2 = tr2S - 2.0 * coef
    # print ("trS2: ", trS2)

    # print ("numer: ", ((1.0 - 2.0 / p) * trS2 + tr2S))
    # print ("denom: ", ((batch_size + 1 - 2.0 / p) * (trS2 - tr2S / p)) )
    rho = min(((1.0 - 2.0 / p) * trS2 + tr2S) / ((batch_size + 1 - 2.0 / p) * (trS2 - tr2S / p)), 1.0)
    diag_shrunk = trS / p

    return float(rho), float(diag_shrunk)


def estimate_shrinkage(eigvals, p, batch_size):
    # p = len(eigvals)
    # Tr(s) = Sum(\lambda_i)
    trS = np.sum(eigvals)
    # print ("trS: ", trS)

    # Tr^2(S) - computed knowing that Tr(s) = Sum(eigvals(S))
    tr2S = trS**2.0
    # print ("tr2S: ", tr2S)

    # TODO: optimize this.
    coef = 0.0
    for j in range(len(eigvals)):
        for i in range(j):
            coef += eigvals[j] * eigvals[i]
    trS2 = tr2S - 2.0 * coef
    # print ("trS2: ", trS2)

    # print ("numer: ", ((1.0 - 2.0 / p) * trS2 + tr2S))
    # print ("denom: ", ((batch_size + 1 - 2.0 / p) * (trS2 - tr2S / p)) )

    numer = ((1.0 - 2.0) / p * trS2 + tr2S)
    denom = ((batch_size + 1 - 2.0) / p * (trS2 - tr2S / p))
    rho = min( numer / denom , 1.0)
    diag_shrunk = trS / p

    return float(rho), float(diag_shrunk)
