
import types
import numpy as np
np.set_printoptions(precision=2)


def cg_solve(F,
             b,
             x_0=None,
             cg_iters=10,
             cg_residual_tol=1e-10,
             damping=1e-4):

    x = np.zeros_like(b) if x_0 is None else x_0
    if x_0 is not None:
        hvp_x0 = F(x) + damping * x
        # isinstance(F, types.FunctionType)
        # hvp_x0 = F @ x + damping * x

    r = b.copy() if x_0 is None else b-hvp_x0
    p = r.copy()
    rdotr = np.dot(r.T, r)
    directions = []
    for i in range(cg_iters):
        # obj = x.T @ (F @ x) - 2 * b.T @ x
        obj = x.T @ F(x) - 2 * b.T @ x
        print ("J: ", obj)
        # hvp_p = F @ p + damping * p
        hvp_p = F(p) + damping * p
        directions.append(r.copy())
        z = hvp_p
        v = rdotr / np.dot(p.T, z) #p.dot(z)
        x += v * p
        r -= v * z
        newrdotr = np.dot(r.T, r) #r.dot(r)
        mu = newrdotr / rdotr
        p = r + mu * p
        rdotr = newrdotr
        # print ("rdotr tru: ", rdotr)
        # if rdotr < cg_residual_tol:
        #     break

    obj = x.T @ F(x) - 2 * b.T @ x
    print ("J: ", obj)

    nd = len(directions)
    for k in range(nd):
        print ("Norm of direction", k,":", np.linalg.norm(directions[k]))
    M = np.zeros((nd, nd))
    for i in range(nd):
        for j in range(i):
            orth = np.dot(directions[i].T, directions[j])
            M[i,j] = orth
    print (M)

    return x, rdotr

# update residuals but use random directions
def cg_solve_random_new_res_rand_dir(F,
                    b,
                    x_0=None,
                    cg_iters=10,
                    cg_residual_tol=1e-10,
                    damping=1e-4):

    x = np.zeros_like(b) if x_0 is None else x_0
    if x_0 is not None:
        hvp_x0 = F(x) + damping * x

    r = b.copy() if x_0 is None else b-hvp_x0
    p = r.copy()
    rdotr = np.dot(r.T, r)
    rorig = r.copy()
    rnorm = np.linalg.norm(r)
    # print ("Rorig: ", np.linalg.norm(rorig))
    # input("")

    directions = []
    for i in range(cg_iters):
        obj = x.T @ F(x) - 2 * b.T @ x
        print ("J: ", obj)

        # Random update
        d = np.random.randn(A.shape[0], 1)
        d = d / np.linalg.norm(d)
        # print ("Norm of d: ", np.linalg.norm(d))
        # print ("norm of r: ", rnorm)
        # input("")
        directions.append(d)
        z = F(d) + damping * d
        v = np.dot(d.T, r) / np.dot(d.T, z)
        x += v * d
        r = b - F(x)
        rdotr = np.dot(r.T, r)
        rnorm = np.linalg.norm(r)

        if rdotr < cg_residual_tol:
            print ("Breaking: ", rdotr)
            break

    obj = x.T @ F(x) - 2 * b.T @ x
    print ("J: ", obj)

    nd = len(directions)
    M = np.zeros((nd, nd))
    for i in range(nd):
        for j in range(i):
            orth = np.dot(directions[i].T, directions[j])
            M[i,j] = orth
    print (M)

    return x, rdotr

# or use proper dirctions but use original resid
def cg_solve_random_old_res_new_dir(F,
                    b,
                    x_0=None,
                    cg_iters=10,
                    cg_residual_tol=1e-10,
                    damping=1e-4):

    x = np.zeros_like(b) if x_0 is None else x_0
    if x_0 is not None:
        hvp_x0 = F(x) + damping * x

    r = b.copy() if x_0 is None else b-hvp_x0
    p = r.copy()
    rorig = r.copy()
    rdotr = np.dot(r.T, r)
    rdotr_orig = rdotr.copy()
    directions = []
    for i in range(cg_iters):
        obj = x.T @ F(x) - 2 * b.T @ x
        print ("J: ", obj)

        hvp_p = F(p) + damping * p

        z = hvp_p
        v = np.dot(p.T, rorig) / np.dot(p.T, z) #p.dot(z)
        x += v * p
        r -= v * z
        newrdotr = np.dot(r.T, r) #r.dot(r)
        mu = newrdotr / rdotr
        p = r + mu * p
        rdotr = newrdotr

        # if rdotr < cg_residual_tol:
        #     break

    obj = x.T @ F(x) - 2 * b.T @ x
    print ("J: ", obj)

    nd = len(directions)
    M = np.zeros((nd, nd))
    for i in range(nd):
        for j in range(i):
            orth = np.dot(directions[i].T, directions[j])
            M[i,j] = orth
    print (M)

    return x, rdotr


# old res coord dir
def cg_solve_random_old_res_coord_dir(F,
                    b,
                    x_0=None,
                    cg_iters=10,
                    cg_residual_tol=1e-10,
                    damping=1e-4):

    x = np.zeros_like(b) if x_0 is None else x_0
    if x_0 is not None:
        hvp_x0 = F(x) + damping * x

    r = b.copy() if x_0 is None else b-hvp_x0
    p = r.copy()

    # print ("P orgin: ", p)
    rorig = r.copy()
    rdotr = np.dot(r.T, r)
    rdotr_orig = rdotr.copy()

    cg_iters = b.shape[0]
    dirs = np.eye(b.shape[0])
    directions = []
    for i in range(cg_iters):
        obj = x.T @ F(x) - 2 * b.T @ x
        print ("J: ", obj)

        p = dirs[i][:,np.newaxis]
        # print (p)
        hvp_p = F(p) + damping * p

        z = hvp_p
        v = np.dot(p.T, rorig) / np.dot(p.T, z) #p.dot(z)
        x += v * p
        # r -= v * z
        # newrdotr = np.dot(r.T, r) #r.dot(r)
        # mu = newrdotr / rdotr
        # p = r + mu * p
        # rdotr = newrdotr

        if rdotr < cg_residual_tol:
            break
    obj = x.T @ F(x) - 2 * b.T @ x
    print ("J: ", obj)

    nd = len(directions)
    M = np.zeros((nd, nd))
    for i in range(nd):
        for j in range(i):
            orth = np.dot(directions[i].T, directions[j])
            M[i,j] = orth
    print (M)

    return x, rdotr


# or use proper dirctions but use original resid
def cg_solve_random_old_res_random_dir(F,
                    b,
                    x_0=None,
                    cg_iters=10,
                    cg_residual_tol=1e-10,
                    damping=1e-4):

    x = np.zeros_like(b) if x_0 is None else x_0
    if x_0 is not None:
        hvp_x0 = F(x) + damping * x

    r = b.copy() if x_0 is None else b-hvp_x0
    p = r.copy()
    rorig = r.copy()
    rdotr = np.dot(r.T, r)
    rdotr_orig = rdotr.copy()
    directions = []
    for i in range(cg_iters):
        obj = x.T @ F(x) - 2 * b.T @ x
        print ("J: ", obj)

        # Random update
        d = np.random.randn(A.shape[0], 1)
        d = d / np.linalg.norm(d)
        # print ("Norm of d: ", np.linalg.norm(d))
        # print ("norm of r: ", rnorm)
        directions.append(d)
        z = F(d) + damping * d
        v = np.dot(d.T, rorig) / np.dot(d.T, z)
        xnew = x + v * d
        obj = xnew.T @ F(xnew) - 2 * b.T @ xnew
        print ("Jnew: ", obj)
        # x += v * d

    # obj = x.T @ F(x) - 2 * b.T @ x
    # print ("J: ", obj)
    #
    # nd = len(directions)
    # M = np.zeros((nd, nd))
    # for i in range(nd):
    #     for j in range(i):
    #         orth = np.dot(directions[i].T, directions[j])
    #         M[i,j] = orth
    # print (M)

    return x, rdotr


if __name__ == '__main__':
    d = 100
    # A = np.eye(d)
    A = np.random.randn(d, 10)
    A = A @ A.T
    # print (A.shape)
    b = np.random.randn(d, 1)

    def mv(v):
        return A @ (A.T @ v)

    print ("True")
    # x_min, x_resid = cg_solve(A, b)
    x_min, x_resid = cg_solve(mv, b)
    print (x_min.shape, x_resid)

    print ("Random A (new residual, random dir)")
    # x_min_rand, x_resid_rand = cg_solve_random_new_res_rand_dir(A, b)
    x_min_rand, x_resid_rand = cg_solve_random_new_res_rand_dir(mv, b)
    print (x_min_rand.shape, x_resid_rand)

    print ("Random B (old residual, new dir)")
    # # x_min_rand, x_resid_rand = cg_solve_random_old_res_new_dir(A, b)
    x_min_rand, x_resid_rand = cg_solve_random_old_res_new_dir(mv, b)
    # print (x_min_rand.shape, x_resid_rand)

    print ("Random C (old residual, coord dir)")
    x_min_rand, x_resid_rand = cg_solve_random_old_res_coord_dir(mv, b)

    print ("Random D (old residtual, random dir)")
    x_min_rand, x_resid_rand = cg_solve_random_old_res_random_dir(mv, b)
