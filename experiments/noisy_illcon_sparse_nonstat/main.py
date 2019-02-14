"""
Generate random (possibly ill-conditioned) quadratic
Generate noisy data from random quadratic
* possibly drop gradients
* possibly make quadratic non stationary

"""

import numpy as np
from scipy.stats import ortho_group

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

def generate_data(dim, noise, samples=1000):
    xstar = np.random.randn(dim)
    data = np.empty((samples, dim))
    for s in range(samples):
        xi = np.random.normal(loc=xstar, scale=noise**2.0)
        data[s,:] = xi
    return xstar, data

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Random quadratic generator')
    parser.add_argument('--dimension', type=int, default=100,
                        help='dimension of the quadratic (default: 100)')
    parser.add_argument('--condition', type=float, default='1.0',
                        help='Approximate conditioning (default: 1.0)')
    parser.add_argument('--noise', type=float, default=0.0,
                        help='noise on samples (default: 0)')
    parser.add_argument('--grad-sparsity', type=float, default=0.0,
                        help='gradient sparsity, g_i is set to zero with this probability (default: 0.0)')
    parser.add_argument('--rotate', action='store_true', default=False,
                        help='Rotate the matrix randomly')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-dir', type=str, default='results/',
                        help='dir to save output')

    args = parser.parse_args()

    np.random.seed(args.seed)

    w = random_eigs_with_condition(args.condition, args.dimension)
    R = random_rotation(args.dimension) if args.rotate else np.eye(args.dimension)
    A = R @ np.diag(w) @ R.T

    xstar, data = generate_data(args.dimension, args.noise)
    print (xstar)
    print (data.shape)

    print (data[0])
    print (data[1])
