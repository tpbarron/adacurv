
import torch

def cg_solve(Fvp_fn,
             b,
             x_0=None,
             cg_iters=10,
             cg_residual_tol=1e-10,
             damping=1e-4,
             shrunk=False,
             rho=None,
             Dshrunk=None):
    """

    """
    if shrunk:
        assert rho is not None
        assert Dshrunk is not None

    x = torch.zeros_like(b) if x_0 is None else x_0
    if x_0 is not None:
        if shrunk:
            hvp_x0 = (1.0 - rho) * Fvp_fn(x) + rho * x * Dshrunk
        else:
            hvp_x0 = Fvp_fn(x) + damping * x


    r = b.clone() if x_0 is None else b-hvp_x0.data
    p = r.clone()
    rdotr = r.dot(r)

    for i in range(cg_iters):
        if shrunk:
            hvp_p = (1.0 - rho) * Fvp_fn(p) + rho * p * Dshrunk
        else:
            hvp_p = Fvp_fn(p) + damping * p

        z = hvp_p.data
        v = rdotr / p.dot(z)
        x += v * p
        r -= v * z
        newrdotr = r.dot(r)
        mu = newrdotr / rdotr
        p = r + mu * p
        rdotr = newrdotr

        if rdotr < cg_residual_tol:
            break
    return x
