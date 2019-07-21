import numpy as np
from scipy.linalg import eigvalsh_tridiagonal

def compute_fvp(Fvp_fn, v, damping):
    hvp = Fvp_fn(v) + damping * v
    return hvp


def cg_solve(Fvp_fn,
             b,
             x_0=None,
             M=None,
             cg_iters=10,
             cg_residual_tol=1e-10,
             damping=1e-4,
             extract_tridiag=False):
    """
    Solve the system Fx = b

    M: diagonal preconditioner in vector form.
    damping: regularize the system and solve Fx + damping I = b
    shrunk: whether to use shrinkage "instead" of damping
    rho: shrinkage factor, required if shrunk is True
    Dshrunk: diagonal covariance to shrink toward, required if shrunk is True
    extract_tridiag: whether to build the tridiagonal Lanczos matrix of dimension (cg_iters x cg_iters)
        that may be used to estimate eigenvalues of F.
        If True will return (x, T)
    """
    if M is not None:
        # M must be a vector (diagonal matrix) of dim equal to b
        assert len(M) == len(b)

    x = np.zeros_like(b) if x_0 is None else x_0
    if x_0 is not None:
        hvp_x0 = compute_fvp(Fvp_fn, x, damping)

    if extract_tridiag:
        diag_elems = []
        off_diag_elems = []
        alpha_prev = 0
        beta_prev = 0

    r = b.copy() if x_0 is None else b-hvp_x0
    if M is not None:
        p = 1.0 / M * r.copy()
    else:
        p = r.copy()
    rdotr = p.dot(r)

    for i in range(cg_iters):
        hvp_p = compute_fvp(Fvp_fn, p, damping)
        z = hvp_p.data
        v = rdotr / p.dot(z)
        x += v * p
        r -= v * z

        if M is not None:
            s = 1.0 / M * r
        else:
            s = r
        newrdotr = s.dot(r)
        mu = newrdotr / rdotr

        if extract_tridiag:
            v = rdotr / p.dot(z)

            alpha = v
            beta = mu
            term1 = 1.0/alpha
            term2 = 0 if i == 0 else beta_prev / alpha_prev

            diag_elems.append(term1 + term2)
            if i < cg_iters-1:
                off_diag_elems.append(np.sqrt(beta) / alpha)
            alpha_prev = alpha
            beta_prev = beta

        p = s + mu * p
        rdotr = newrdotr

        if rdotr < cg_residual_tol:
            break
    if extract_tridiag:
        off_diag_elems = off_diag_elems[0:len(diag_elems)-1]
        # print ("CG diag: ", diag_elems)
        # print ("CG diag_adj: ", off_diag_elems)
        return x, (np.array(diag_elems), np.array(off_diag_elems))
    return x


def make_fvp_fn(A):
    def f(v):
        return A @ v
    return f

if __name__ == "__main__":

    n = 100
    P = np.random.random((100, 100))
    A = P @ P.T
    M = np.diag(A)
    Minv_mat = np.diag(1.0/M)

    w1 = np.linalg.eigvals(A)
    w1b = np.linalg.eigvals(Minv_mat @ A)

    b = np.ones((n,))
    fvp_fn = make_fvp_fn(A)

    cg_result = cg_solve(fvp_fn, b, cg_iters=n, extract_tridiag=True)
    ng, (diag_elems, off_diag_elems) = cg_result
    w2 = eigvalsh_tridiagonal(diag_elems, off_diag_elems)

    cg_result = cg_solve(fvp_fn, b, cg_iters=n, M=M, extract_tridiag=True)
    ng, (diag_elems, off_diag_elems) = cg_result
    w3 = eigvalsh_tridiagonal(diag_elems, off_diag_elems)
    w4 = Minv_mat @ np.diag(w3)

    print ("Originals: ", np.max(w1), np.linalg.norm(w1))
    print ("CG no prec: ", np.max(w2), np.linalg.norm(w2), np.max(w1)-np.max(w2), np.linalg.norm(w1-w2))
    print ("CG w/ prec: ", np.max(w3), np.linalg.norm(w3), np.max(w1)-np.max(w3), np.linalg.norm(w1-w3))
    print ("CG w/ prec vs orig: ", np.max(w4), np.linalg.norm(w4), np.max(w1)-np.max(w4), np.linalg.norm(w1-w4))
    print ("CG w/ prec vs true MinvA: ", np.max(w4), np.linalg.norm(w4), np.max(w1b)-np.max(w4), np.linalg.norm(w1b-w4))
