
def cg_solve(Fvp_fn,
             b,
             x_0=None,
             M=None,
             cg_iters=10,
             cg_residual_tol=1e-10,
             damping=1e-4):
    """
    Solve the system Fx = b

    M: diagonal preconditioner in vector form.
    damping: regularize the system and solve Fx + damping I = b
    """
    x = torch.zeros_like(b) if x_0 is None else x_0
    if x_0 is not None:
        hvp_x0 = compute_fvp(Fvp_fn, x, damping, shrunk, rho, Dshrunk)
        cg_data[0]['hvp'] = hvp_x0.numpy()

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
        cg_data[i+1] = {}

        hvp_p = compute_fvp(Fvp_fn, p, damping, shrunk, rho, Dshrunk)
        z = hvp_p.data
        cg_data[i+1]['hvp'] = z.numpy()

        v = rdotr / p.dot(z)
        # v_ = rdotr_ / p_.dot(z)
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
            # v = r_t M r_t / r_t M F p_t
            # mu = r_t+1 M r_t+1 / r_t M r_t
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

        p = s + mu * p
        rdotr = newrdotr

        cg_data[i+1]['x'] = x.numpy()
        cg_data[i+1]['residual'] = r.numpy()
        cg_data[i+1]['direction'] = p.numpy()
        cg_data[i+1]['residual_norm'] = rdotr.numpy()

        if rdotr < cg_residual_tol:
            break
    if extract_tridiag:
        off_diag_elems = off_diag_elems[0:len(diag_elems)-1]
        # print ("CG diag: ", diag_elems)
        # print ("CG diag_adj: ", off_diag_elems)
        return dict(x=x, diag=(np.array(diag_elems), np.array(off_diag_elems)), cg_log=cg_data)
    return dict(x=x, cg_log=cg_data)

def cg_numpy()

if __name__ == "__main__":
