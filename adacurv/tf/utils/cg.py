import tensorflow as tf

from adacurv.tf.utils import curvature_matrix_vector_products as cmvps

def initialize_cg_vars(model_vars):
    # CG variables
    dtype = tf.float32
    with tf.name_scope('cg_vars'):
        cg_step = tf.Variable(0, trainable=False, dtype=tf.int32)

        cg_delta = []
        directions = []
        residuals = []
        residual_norm = tf.Variable(0, trainable=False, dtype=tf.float32)

        for w in model_vars:
            zeros = tf.zeros(w.get_shape(), dtype=dtype)
            delta = tf.Variable(zeros, dtype=dtype, name='delta')
            cg_delta.append(delta)
            d = tf.Variable(zeros, dtype=dtype, name='direction')
            directions.append(d)
            r = tf.Variable(zeros, dtype=dtype, name='residual')
            residuals.append(r)

    return cg_step, cg_delta, directions, residuals, residual_norm

def create_reset_cg_delta_op(cg_delta, cg_decay):
    update_delta_0_ops = []
    for delta in cg_delta:
        update_delta = tf.assign(delta, cg_decay * delta)
        update_delta_0_ops.append(update_delta)
    update_delta_0_op = tf.group(*update_delta_0_ops)
    return update_delta_0_op

def create_reset_cg_step_op(cg_step):
    reset_cg_step = tf.assign(cg_step, 0)
    return reset_cg_step


def cg_solve(Cvp_fn,
             gradients,
             loss,
             z,
             vars,
             cg_step,
             cg_delta,
             directions,
             residuals,
             residual_norm,
             cg_iters=10,
             bias2_correction=1.0):

    def cg_iter(gradients):
        with tf.name_scope('conjugate_gradient'):
            # Ax = cmvps.Fvp(loss, vars, cg_delta)
            Ax = Cvp_fn(loss, z, vars, cg_delta, damping=0.1)
            # with tf.control_dependencies([tf.print(ax) for ax in Ax]):
            Ax = [ax / bias2_correction for ax in Ax]

            b = gradients #[grad for grad in gradients]
            bAx = [tf.subtract(Ax, b) for b, Ax in zip(b, Ax)]

            condition = tf.equal(cg_step, 0)

            r = [tf.cond(condition, lambda: tf.assign(r,  bax), lambda: r) for r, bax  in zip(residuals, bAx)]
            # r = tf.cond(condition, lambda: tf.assign(residual,  Avar @ delta - bvar), lambda: residual)
            with tf.control_dependencies(r):
                d = [tf.cond(condition, lambda: tf.assign(d, -r), lambda: d) for  d, r in zip(directions, r)]
                # d = tf.cond(condition, lambda: tf.assign(direction, -residual), lambda: direction)

                Ad = Cvp_fn(loss, z, vars, d, damping=0.1)

                residual_norm = tf.reduce_sum([tf.reduce_sum(r**2) for r in r])
                # residual_norm = tf.reduce_sum(r**2)

                alpha = residual_norm / tf.reduce_sum([tf.reduce_sum(d * ad) for d, ad in zip(d, Ad)])
                # alpha = residual_norm / tf.reduce_sum(d * Ad)

                beta = tf.reduce_sum([tf.reduce_sum((r + alpha * ad)**2) for r, ad in zip(r, Ad)]) / residual_norm
                # beta = tf.reduce_sum((r + alpha * Ad)**2) / residual_norm

                cg_delta_update_ops = []
                cg_directions_update_ops = []
                cg_residuals_update_ops = []
                for i, delta  in list(enumerate(cg_delta)):
                    update_delta = tf.assign(delta, delta + alpha * d[i], name='update_delta')
                    update_residual = tf.assign(residuals[i], r[i] + alpha * Ad[i], name='update_residual')
                    update_direction = tf.assign(directions[i], beta * d[i] - (r[i] + alpha * Ad[i]), name='update_direction')

                    cg_delta_update_ops.append(update_delta)
                    cg_directions_update_ops.append(update_direction)
                    cg_residuals_update_ops.append(update_residual)
                # update_delta = tf.assign(delta, delta + alpha * d, name='update_delta')
                # update_residual = tf.assign(residual, r + alpha * Ad, name='update_residual')
                # update_direction = tf.assign(direction, beta * d - (r + alpha * Ad), name='update_direction')
                cg_step_up = tf.assign_add(cg_step, 1)

        return cg_step_up, cg_delta_update_ops, cg_directions_update_ops, cg_residuals_update_ops, residual_norm
        # return cg_step_up, update_delta, update_residual, update_direction, residual_norm

    def cg_cond(cg_step, cg_delta, cg_directions, cg_residuals, residual_norm, gradients):
        return tf.less(cg_step, cg_iters)

    def cg_body(cg_step, cg_delta, cg_directions, cg_residuals, residual_norm, gradients):
        cg_step, cg_delta_update_ops, cg_directions_update_ops, cg_residuals_update_ops, residual_norm = cg_iter(gradients)
        return cg_step, cg_delta_update_ops, cg_directions_update_ops, cg_residuals_update_ops, residual_norm, gradients

    cg_op = tf.while_loop(
        cg_cond,
        cg_body,
        (cg_step, cg_delta, directions, residuals, residual_norm, gradients),
        parallel_iterations=1)
    return cg_op
