# -*- coding: utf-8 -*-

import numpy as np
from scipy.sparse.linalg import cg
import tensorflow as tf
import time

np.random.seed(3)
tf.random.set_random_seed(3)

def conjugate_grad(A, b, x=None):
    """
    Description
    -----------
    Solve a linear equation Ax = b with conjugate gradient method.

    Parameters
    ----------
    A: 2d numpy.array of positive semi-definite (symmetric) matrix
    b: 1d numpy.array
    x: 1d numpy.array of initial point

    Returns
    -------
    1d numpy.array x such that Ax = b
    """
    n = len(b)
    if not x:
        x = np.ones(n)
    r = np.dot(A, x) - b
    print ("r init: ", r[0:5])
    p = - r
    r_k_norm = np.dot(r, r)
    for i in range(2*n): #50):
        Ap = np.dot(A, p)
        alpha = r_k_norm / np.dot(p, Ap)
        x += alpha * p
        r += alpha * Ap
        r_kplus1_norm = np.dot(r, r)
        print ("r new: ", r_k_norm, r_kplus1_norm, r[0:5])
        beta = r_kplus1_norm / r_k_norm
        r_k_norm = r_kplus1_norm
        if r_kplus1_norm < 1e-5:
            print ('Itr:', i)
            break
        p = beta * p - r
    return x

def run():

    n = 10
    P = np.random.normal(size=[n, n])
    A = np.dot(P.T, P)
    b = np.ones(n)

    t1 = time.time()
    print ('start')
    x = conjugate_grad(A, b)
    t2 = time.time()
    print (t2 - t1)
    x2 = np.linalg.solve(A, b)
    t3 = time.time()
    print (t3 - t2, np.linalg.norm(x - x2))
    x3 = cg(A, b)[0]
    t4 = time.time()
    print (t4 - t3, np.linalg.norm(x - x3))

    import tensorflow as tf
    dtype = tf.float64
    with tf.name_scope('cg_vars'):
        Avar = tf.Variable(A, dtype=dtype)
        bvar = tf.Variable(b.reshape(n, 1), dtype=dtype)
        cg_step = tf.Variable(0, trainable=False, dtype=tf.int32)
        dl = tf.Variable(0, trainable=False, dtype=dtype)

        zeros = tf.zeros((n,1), dtype=dtype)
        ones = tf.ones((n,1), dtype=dtype)

        delta = tf.Variable(ones, dtype=dtype, name='delta')
        direction = tf.Variable(zeros, dtype=dtype, name='direction')
        residual = tf.Variable(zeros, dtype=dtype, name='residual')
        residual_norm = tf.Variable(0, trainable=False, dtype=dtype)

    print ("cg step 1: ", cg_step)
    reset_cg_step = tf.assign(cg_step, 0)

    # with tf.name_scope('conjugate_gradient'):
    #     # If first step, set to bAx, r
    #     condition = tf.equal(cg_step, 0)
    #
    #     r = tf.cond(condition, lambda: tf.assign(residual,  Avar @ delta - bvar), lambda: residual)
    #     with tf.control_dependencies([r]):
    #         d = tf.cond(condition, lambda: tf.assign(direction, -residual), lambda: direction)
    #         # with tf.control_dependencies([d]):
    #         residual_norm = tf.reduce_sum(r**2)
    #
    #         Ad = Avar @ d
    #
    #         alpha = residual_norm / tf.reduce_sum(d * Ad)
    #         beta = tf.reduce_sum((r + alpha * Ad)**2) / residual_norm
    #
    #         update_delta = tf.assign(delta, delta + alpha * d, name='update_delta')
    #         update_residual = tf.assign(residual, r + alpha * Ad, name='update_residual')
    #         update_direction = tf.assign(direction, beta * d - (r + alpha * Ad), name='update_direction')

    print ("cg step 2: ", cg_step)

    def cg_iter():
        with tf.name_scope('conjugate_gradient'):

            condition = tf.equal(cg_step, 0)

            r = tf.cond(condition, lambda: tf.assign(residual,  Avar @ delta - bvar), lambda: residual)
            with tf.control_dependencies([r]):
                d = tf.cond(condition, lambda: tf.assign(direction, -residual), lambda: direction)
                residual_norm = tf.reduce_sum(r**2)

                Ad = Avar @ d

                alpha = residual_norm / tf.reduce_sum(d * Ad)
                beta = tf.reduce_sum((r + alpha * Ad)**2) / residual_norm

                update_delta = tf.assign(delta, delta + alpha * d, name='update_delta')
                update_residual = tf.assign(residual, r + alpha * Ad, name='update_residual')
                update_direction = tf.assign(direction, beta * d - (r + alpha * Ad), name='update_direction')

                cg_step_up = tf.assign_add(cg_step, 1)

        return cg_step_up, update_delta, update_residual, update_direction, residual_norm

    def cg_cond(cg_step, cg_delta, cg_directions, cg_residuals, residual_norm):
        return tf.less(cg_step, 2*n)

    def cg_body(cg_step, cg_delta, cg_directions, cg_residuals, residual_norm):
        # with tf.control_dependencies([tf.print(tf.equal(cg_step, 0))]):
        cg_step, cg_delta_update_ops, cg_directions_update_ops, cg_residuals_update_ops, residual_norm = cg_iter()
        return cg_step, cg_delta_update_ops, cg_directions_update_ops, cg_residuals_update_ops, residual_norm

    with tf.control_dependencies([reset_cg_step]):
        # gradients = tf.gradients(loss, vars)
        cg_op = tf.while_loop(
            cg_cond,
            cg_body,
            (cg_step, delta, direction, residual, residual_norm),
            back_prop=False,
            parallel_iterations=1)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        step, dlt, dir, res, res_norm = sess.run(cg_op)
        print (step, res_norm)
        print ("dist: ", np.linalg.norm(x - dlt[:,0]))

        # for i in range(2*n):
        #     # res_norm, res = sess.run([residual_norm, r])
        #     dlt, res_up, dir = sess.run([update_delta, update_residual, update_direction])
        #     itr = sess.run(cg_step)
        #     sess.run(tf.assign_add(cg_step, 1))
        #     print (itr, ":, res, dist: ", res_up, np.linalg.norm(x - dlt[:,0]))
        #
        # x4 = sess.run(delta)
        # print ("dist: ", np.linalg.norm(x - x4[:,0]))

if __name__ == '__main__':
    run()
