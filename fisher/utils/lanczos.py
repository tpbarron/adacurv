import numpy as np

import torch
from torch.nn.utils import parameters_to_vector

from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import eigsh

def lanczos_iteration(Fvp_fn, params, k=20):
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

def estimate_shrinkage(eigvals, p, batch_size):
    # Tr(s) = Sum(\lambda_i)
    trS = np.sum(eigvals)

    # Tr^2(S) - computed knowing that Tr(s) = Sum(eigvals(S))
    tr2S = trS**2.0

    # TODO: optimize this.
    coef = 0.0
    for j in range(len(eigvals)):
        for i in range(j):
            coef += eigvals[j] * eigvals[i]
    trS2 = tr2S - 2.0 * coef

    rho = min(((1.0 - 2.0 / p) * trS2 + tr2S) / ((batch_size + 1 - 2.0 / p) * (trS2 - tr2S / p)), 1.0)
    diag_shrunk = trS / p

    return float(rho), float(diag_shrunk)
