
import numpy as np
import torch


def compute_fvp(Fvp_fn, v, damping, shrunk, rho, Dshrunk):
    if shrunk:
        hvp = (1.0 - rho) * Fvp_fn(v) + rho * v * Dshrunk
    else:
        hvp = Fvp_fn(v) + damping * v
    return hvp

def cg_solve(Fvp_fn,
             b,
             x_0=None,
             M=None,
             cg_iters=10,
             cg_residual_tol=1e-10,
             damping=1e-4,
             shrunk=False,
             rho=None,
             Dshrunk=None,
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
        # M must be a vector (diagon matrix) of dim equal to b
        assert len(M) == len(b)

    if shrunk:
        assert rho is not None
        assert Dshrunk is not None

    x = torch.zeros_like(b) if x_0 is None else x_0
    if x_0 is not None:
        hvp_x0 = compute_fvp(Fvp_fn, x, damping, shrunk, rho, Dshrunk)

    if extract_tridiag:
        diag_elems = []
        off_diag_elems = []
        alpha_prev = 0
        beta_prev = 0

    r = b.clone() if x_0 is None else b-hvp_x0.data
    if M is not None:
        p = 1.0 / M * r.clone()
    else:
        p = r.clone()
    rdotr = p.dot(r)

    for i in range(cg_iters):
        hvp_p = compute_fvp(Fvp_fn, p, damping, shrunk, rho, Dshrunk)
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
        p = s + mu * p
        rdotr = newrdotr

        if extract_tridiag:
            alpha = v
            beta = mu
            term1 = 1.0/alpha
            term2 = 0 if i == 0 else beta_prev / alpha_prev
            # T[i,i] = term1 + term2
            # if i < cg_iters-1:
            #     T[i,i+1] = T[i+1,i] = np.sqrt(beta) / alpha

            diag_elems.append(term1 + term2)
            if i < cg_iters-1:
                off_diag_elems.append(np.sqrt(beta) / alpha)
            alpha_prev = alpha
            beta_prev = beta

        if rdotr < cg_residual_tol:
            break
    if extract_tridiag:
        off_diag_elems = off_diag_elems[0:len(diag_elems)-1]
        return x, (np.array(diag_elems), np.array(off_diag_elems))
    return x
