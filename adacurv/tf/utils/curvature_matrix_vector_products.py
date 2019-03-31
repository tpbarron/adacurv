
import tensorflow as tf

# from tensorflow_forward_ad.second_order import fisher_vec_bk, hessian_vec_bk
from adacurv.tf.utils import fwd_gradients

def fisher_vec_fw(ys, xs, vs):
    """Implements Fisher vector product using backward and forward AD.
    Args:
        ys: Loss function or output variables.
        xs: Weights, list of tensors.
        vs: List of tensors to multiply, for each weight tensor.
    Returns:
        J'Jv: Fisher vector product.
    """
    # Validate the input
    if type(xs) == list:
        if len(vs) != len(xs):
            raise ValueError("xs and vs must have the same length.")
    if type(ys) != list:
        ys = [ys]
    jv = fwd_gradients.forward_gradients(ys, xs, vs)
    jjv = tf.gradients(ys, xs, jv, gate_gradients=True)
    return jjv

def gauss_newton_vec(ys, zs, xs, vs):
    """Implements Gauss-Newton vector product.
    Args:
        ys: Loss function.
        zs: Before output layer (input to softmax).
        xs: Weights, list of tensors.
        vs: List of perturbation vector for each weight tensor.
    Returns:
        J'HJv: Guass-Newton vector product.
    """
    # Validate the input
    if type(xs) == list:
        if len(vs) != len(xs):
            raise ValueError("xs and vs must have the same length.")
    grads_z = tf.gradients(ys, zs, gate_gradients=True)
    hjv = fwd_gradients.forward_gradients(grads_z, xs, vs)
    jhjv = tf.gradients(zs, xs, hjv, gate_gradients=True)
    return jhjv #, hjv

def Fvp(loss, z, vars, vecs, damping=0.0):
    Fv = fisher_vec_fw(loss, vars, vecs)
    damped_fvp = [tf.add(Fv[i], tf.multiply(damping, tf.ones(Fv[i].shape))) for i in range(len(Fv))]
    return damped_fvp

def GNvp(loss, z, vars, vecs, damping=0.0):
    Fv = gauss_newton_vec(loss, z, vars, vecs)
    damped_fvp = [tf.add(Fv[i], tf.multiply(damping, tf.ones(Fv[i].shape))) for i in range(len(Fv))]
    return damped_fvp
