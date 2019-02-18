"""
Generate random (possibly ill-conditioned) quadratic
Generate noisy data from random quadratic
* possibly drop gradients
* possibly make quadratic non stationary

"""

import numpy as np
from scipy.stats import ortho_group

###
# Utils to generate quadratic
###

def perform_induction_step(M):
    d = M.shape[0]
    dnew = d+1
    Mnew = np.eye(dnew)
    Mnew[1:,1:] = M

    v = np.random.randn(dnew)[:,np.newaxis]
    v /= np.linalg.norm(v)

    H = np.eye(dnew) - 2.0 * v @ v.T
    Mnew = H @ Mnew
    return Mnew

def random_rotation(dim):
    th = np.random.uniform(low=0.0, high=2.0*np.pi)
    M = np.array([[np.cos(th), -np.sin(th)],
                  [np.sin(th), np.cos(th)]])
    while M.shape[0] < dim:
        # print (M)
        # input("")
        M = perform_induction_step(M)
    return M
    # print (M)

def generate_random_matrix_with_eigs(w):
    Q = ortho_group.rvs(len(w))
    D = np.diag(w)
    A = Q.T @ D @ Q
    return A

def random_eigs_with_condition(cond, dim):
    if cond == 1.0:
        w = np.random.uniform(low=0.1, high=1.1, size=(dim,))
    else:
        d1 = int(0.9 * dim)
        d2 = dim - d1
        w1 = np.random.uniform(low=0.0, high=1.0, size=(d1,))
        w2 = np.random.uniform(low=cond/2.0, high=cond, size=(d2,))
        w = np.concatenate((w1, w2))
    return w

def generate_quadratic(cond, dim, rotate):
    w = random_eigs_with_condition(cond, dim)
    R = random_rotation(dim) if rotate else np.eye(dim)
    A = R @ np.diag(w) @ R.T
    return A
