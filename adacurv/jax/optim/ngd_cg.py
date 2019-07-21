
import jax.numpy as np
from jax.experimental.optimizers import make_schedule, optimizer

@optimizer
def ngd_cg(step_size, b1=0.9, b2=0.999, eps=1e-8, lmda=0.001, decay=0.9):
    """Construct optimizer triple for Adam.
        Args:
        step_size: positive scalar, or a callable representing a step size schedule
          that maps the iteration index to positive scalar.
        b1: optional, a positive scalar value for beta_1, the exponential decay rate
          for the first moment estimates (default 0.9).
        b2: optional, a positive scalar value for beta_2, the exponential decay rate
          for the second moment estimates (default 0.999).
        eps: optional, a positive scalar value for epsilon, a small constant for
          numerical stability (default 1e-8).
        Returns:
        An (init_fun, update_fun, get_params) triple.
    """
    step_size = make_schedule(step_size)
    def init(x0):
        return x0,
    def update(i, g, state):
        # Get gradients
        # Solve cg
        ng = cg_solve(Fvp_fn, g)

        # compute step size based on stats
        lr = step_size(i)
        alpha = np.sqrt(np.abs(lr / (np.dot(g, ng) + 1e-20)))

        # update params
        x = x - alpha * ng
        return x
    def get_params(state):
        x, = state
        return x
    return init, update, get_params
