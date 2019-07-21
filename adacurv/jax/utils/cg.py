
import jax.numpy as jnp
import jax.tree_util as tu
from jax.flatten_util import ravel_pytree

def _tree_ones_like(tree):
  def f(x):
    return jnp.ones_like(x)
  return tu.tree_map(f, tree)

def _tree_dot(t1, t2):
    assert(len(t1) == len(t2))
    def f(x, y):
        return x + y

    return tu.tree_reduce(lambda x, y: x+y, tu.tree_multimap(f, t1, t2)) #jnp.dot(tu.tree_flatten(t1), tu.tree_flatten(t2))

def _tree_copy(tree):
    def f(x):
        arr = jnp.empty_like(x)
        arr.fill(x)
        return arr
    return tu.tree_map(f, tree)

def cg_solve_jax_hvp(hvp_fn,
                 b,
                 x_0,
                 cg_iters=10,
                 cg_residual_tol=1e-20,
                 damping=1e-4):
    """
    Solve the system Fx = b
    damping: regularize the system and solve Fx + damping I = b
    """
    x = jnp.zeros_like(b)
    # x = x_0 #jnp.zeros_like(b) #if x_0 is None else x_0 #, copy=True) if x_0 is None else x_0
    # hvp_x0 = hvp_fn(x) + damping * x #jnp.dot(A, x)

    r = jnp.array(b, copy=True)
    # r = b - hvp_x0
    p = jnp.array(r, copy=True)
    rdotr = p.dot(r)

    for i in range(cg_iters):
        z = hvp_fn(p) + damping * p

        v = rdotr / p.dot(z)
        x += v * p
        r -= v * z

        s = r
        newrdotr = s.dot(r)
        mu = newrdotr / rdotr

        p = s + mu * p
        rdotr = newrdotr

        # Note: ignoring residual tol because we don't reach it and it makes this function not jit-able.
        # if rdotr < cg_residual_tol:
        #     break
    return x


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
