
import tensorflow as tf

def forward_gradients(ys, xs, grad_xs=None, stop_gradients=None,
                  colocate_gradients_with_ops=True):
    """Compute forward-mode gradients."""
    # See b/37888268.

    # This version of forward-mode autodiff is based on code by Tim Cooijmans
    # and handles list arguments and certain special cases such as when the
    # ys doesn't depend on one or more of the xs, and when tf.IndexedSlices are
    # generated by the first tf.gradients call.

    us = [tf.zeros_like(y) + float("nan") for y in ys]
    dydxs = tf.gradients(ys, xs, grad_ys=us, stop_gradients=stop_gradients,
                       colocate_gradients_with_ops=colocate_gradients_with_ops)

    # Deal with strange types that tf.gradients returns but can't
    # deal with.
    dydxs = [
      tf.convert_to_tensor(dydx) if isinstance(dydx, tf.IndexedSlices) else dydx
      for dydx in dydxs
    ]
    dydxs = [
      tf.zeros_like(x) if dydx is None else dydx for x, dydx in zip(xs, dydxs)
    ]
    dysdx = tf.gradients(dydxs, us, grad_ys=grad_xs,
                       colocate_gradients_with_ops=colocate_gradients_with_ops)
    return dysdx
