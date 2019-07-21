
import unittest

from scipy.sparse.linalg import cg
import numpy as np
np.random.seed(3)

def cg_solve_npy(A,
                 b,
                 x_0=None,
                 cg_iters=10,
                 cg_residual_tol=1e-10,
                 damping=1e-4):
    """
    Solve the system Fx = b
    damping: regularize the system and solve Fx + damping I = b
    """
    x = np.zeros_like(b) if x_0 is None else x_0
    if x_0 is not None:
        hvp_x0 = np.dot(A, x)

    r = b.copy() if x_0 is None else b-hvp_x0
    p = r.copy()
    rdotr = p.dot(r)

    for i in range(cg_iters):
        hvp_p = np.dot(A, p)
        z = hvp_p

        v = rdotr / p.dot(z)
        x += v * p
        r -= v * z

        s = r
        newrdotr = s.dot(r)
        mu = newrdotr / rdotr

        p = s + mu * p
        rdotr = newrdotr

        if rdotr < cg_residual_tol:
            break
    return x


from jax.config import config
# CG loses precision with floating point.
config.update("jax_enable_x64", True)
import jax.numpy as jnp

def cg_solve_jax(A,
                 b,
                 x_0=None,
                 cg_iters=10,
                 cg_residual_tol=1e-20,
                 damping=1e-4):
    """
    Solve the system Fx = b
    damping: regularize the system and solve Fx + damping I = b
    """
    x = jnp.zeros_like(b) if x_0 is None else x_0
    if x_0 is not None:
        hvp_x0 = jnp.dot(A, x)

    r = b.copy() if x_0 is None else b-hvp_x0
    p = r.copy()
    rdotr = p.dot(r)

    for i in range(cg_iters):
        hvp_p = jnp.dot(A, p)
        z = hvp_p

        v = rdotr / p.dot(z)
        x += v * p
        r -= v * z

        s = r
        newrdotr = s.dot(r)
        mu = newrdotr / rdotr

        p = s + mu * p
        rdotr = newrdotr

        if rdotr < cg_residual_tol:
            break
    return x


class ConjugateGradientTestCase(unittest.TestCase):

    def setUp(self):
        self.A = np.random.random((10, 10))
        self.A = np.dot(self.A.T, self.A)
        self.b = np.random.random((10,))
        self.x_scipy = cg(self.A, self.b, atol=1e-10)[0]

    def test_scipy_cg_valid(self):
        self.assertTrue(np.allclose(np.dot(self.A, self.x_scipy), self.b))

    def test_numpy_cg_equal(self):
        x_npy = cg_solve_npy(self.A, self.b, cg_iters=1000)
        self.assertTrue(np.allclose(self.x_scipy, x_npy))

    def test_jax_cg_equal(self):
        x_jax = cg_solve_jax(self.A, self.b, cg_iters=1000)
        self.assertTrue(np.allclose(self.x_scipy, x_jax))

####################################################################################################
#
####################################################################################################

from jax.scipy.stats.norm import logpdf, pdf
from scipy.stats import norm

from jax import grad, vmap, hessian, jacobian

class OneDFIMTestCase(unittest.TestCase):

    def setUp(self):
        self.sigma_1d = np.random.uniform()
        self.z_1d = np.random.random()
        self.y_1d = np.random.random()

    def test_log_normal_deriv_1d(self):
        df_dz = (self.y_1d - self.z_1d) / (self.sigma_1d ** 2.0)
        grad_log_normal = grad(logpdf, argnums=1)
        df_dz_jax = grad_log_normal(self.y_1d, self.z_1d, self.sigma_1d)
        self.assertTrue(np.allclose(df_dz, df_dz_jax), \
            "Analytical grad [" + str(df_dz) + "] and jax grad [" + str(df_dz_jax) + "] are not equal")


class NDFIMTestCase(unittest.TestCase):

    def setUp(self):
        n = 10
        self.sigma_nd = np.random.uniform(size=(n,))
        self.x_nd = np.random.random(size=(n,))
        self.W = np.random.random(size=(n, n))

        self.z_nd = jnp.dot(self.W, self.x_nd)
        self.y_nd = np.random.random(size=(n,))

    def test_log_normal_deriv_nd(self):
        def _mv_log_pdf(y, z, s):
            return jnp.sum(logpdf(y, z, s))
        df_dz = (self.y_nd - self.z_nd) / (self.sigma_nd ** 2.0)
        grad_log_normal = grad(_mv_log_pdf, argnums=1)
        df_dz_jax = grad_log_normal(self.y_nd, self.z_nd, self.sigma_nd)
        self.assertTrue(np.allclose(df_dz, df_dz_jax), \
            "Analytical grad [" + str(df_dz) + "] and jax grad [" + str(df_dz_jax) + "] are not equal")

    def test_predictive_fisher(self):
        # F_r = -E_{R_{y | z}} H_{log r} should equal manual computation
        def _mv_log_pdf(y, z, s):
            return jnp.sum(logpdf(y, z, s))
        # def _fisher(y, z, s):
        #     return jnp.sum(_mv_log_pdf(y, z, s))
        d2f_dz = 1.0 / (self.sigma_nd ** 2.0)
        # fisher_log_normal = grad(_fisher, argnums=1)
        fisher_log_normal = hessian(_mv_log_pdf, argnums=1)
        d2f_dz_jax = -jnp.diag(fisher_log_normal(self.y_nd, self.z_nd, self.sigma_nd))
        self.assertTrue(np.allclose(d2f_dz, d2f_dz_jax), \
            "Analytical grad [" + str(d2f_dz) + "] and jax grad [" + str(d2f_dz_jax) + "] are not equal")

    def test_parameterized_predictive_fisher(self):

        def _mv_log_pdf(y, x, s):
            z = jnp.dot(self.W, x)
            return jnp.sum(logpdf(y, z, s))

        def _fn(W, x):
            return jnp.dot(W, x)

        d2r_dz = 1.0 / (self.sigma_nd ** 2.0)

        jac = jacobian(_fn, argnums=0)
        print (self.W.shape, self.x_nd.shape)
        input("pf1")
        df_dw = jac(self.W, self.x_nd[:,np.newaxis])
        print (df_dw.shape)
        input("pf2")
        df_dw_t = df_dw.transpose()
        print (df_dw.shape, df_dw_t.shape, np.diag(d2r_dz).shape)
        param_fisher = np.dot(df_dw_t, np.dot(np.diag(d2r_dz), df_dw))
        print (df_dw)
        input("")

        fisher_log_normal = hessian(_mv_log_pdf, argnums=1)
        jax_fisher = -(fisher_log_normal(self.y_nd, self.x_nd, self.sigma_nd))

        print (param_fisher, jax_fisher)
        # Verify that hessian is equal
        # Verify that vector products equal
        self.assertTrue(True)


####################################################################################################
#
####################################################################################################

# class GNNTestCase(unittest.TestCase):
#
#     def setUp(self):
#         self.sigma_1d = np.random.uniform()
#         self.z_1d = np.random.random()
#         self.y_1d = np.random.random()
#
#         n = 10
#         self.sigma_nd = np.random.uniform(size=(n,))
#         self.z_nd = np.random.random(size=(n,))
#         self.y_nd = np.random.random(size=(n,))

if __name__ == '__main__':
    unittest.main()
