import numpy as np
import tensorflow as tf
import fashion_mnist_data as mnist
from tensorflow_forward_ad import forward_gradients
from tensorflow_forward_ad.second_order import fisher_vec_bk, hessian_vec_bk, gauss_newton_vec

from adacurv.tf.optimizers import NGDOptimizer

# mnist = tf.keras.datasets.mnist

# (x_train, y_train),(x_test, y_test) = mnist.load_data()
# x_train, x_test = x_train / 255.0, x_test / 255.0

bs = 250

(examples, labels) = mnist.load_mnist_as_iterator(2,
                                                  bs,
                                                  use_fake_data=False,
                                                  flatten_images=False)

# x = tf.placeholder(shape=(bs, 28, 28), dtype='float32', name='input')
# y = tf.placeholder(shape=(bs,), dtype='int64', name='output')

h = tf.keras.layers.Reshape((28, 28, 1), input_shape=(bs, 28, 28))(examples)
h = tf.keras.layers.Conv2D(filters=32, kernel_size=5, activation=tf.nn.relu, use_bias=False)(h)
h = tf.keras.layers.MaxPool2D(pool_size=3, strides=2)(h)
h = tf.keras.layers.Conv2D(filters=64, kernel_size=5, activation=tf.nn.relu, use_bias=False)(h)
h = tf.keras.layers.MaxPool2D(pool_size=3, strides=2)(h)
h = tf.keras.layers.Flatten()(h)
h = tf.keras.layers.Dense(1024, activation=tf.nn.relu, use_bias=False)(h)
z = tf.keras.layers.Dense(10, activation=None, use_bias=False)(h)

pred = tf.nn.log_softmax(z)
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=z))
accuracy = tf.reduce_mean(tf.cast(tf.equal(labels, tf.argmax(z, axis=1)), dtype=tf.float32))

vars = tf.trainable_variables()

def fwd_gradients(ys, xs, d_xs):
    """Forward-mode pushforward analogous to the pullback defined by tf.gradients.
    With tf.gradients, grad_ys is the vector being pulled back, and here d_xs is
    the vector being pushed forward."""
    v = tf.placeholder_with_default(np.zeros(ys.get_shape(), dtype=np.float32), shape=ys.get_shape())  # dummy variable
    g = tf.gradients(ys, xs, grad_ys=v, gate_gradients=True)
    return tf.gradients(g, v, grad_ys=d_xs, gate_gradients=True)

# def Fvp(vecs):
#     Jv = tf.expand_dims(tf.reduce_mean(fwd_gradients(z, vars, vecs)[0], axis=0), 1)
#
#     z_mean = tf.reduce_mean(z, axis=0)
#     z_diag = tf.diag(z_mean)
#     z_mean_expand = tf.expand_dims(z_mean, 1)
#     z_outer = tf.matmul(z_mean_expand, tf.transpose(z_mean_expand))
#     F = z_diag - z_outer
#     FJv = tf.matmul(F, Jv)
#
#     JFJv = tf.gradients(z_mean, vars, grad_ys=FJv, gate_gradients=True)
#     return JFJv

def Fvp(vecs):
    Fv = fisher_vec_bk(pred, vars, vecs)
    # Fv = hessian_vec_bk(loss, vars, vecs)
    # Fv = gauss_newton_vec(loss, z, vars, vecs)
    # print (Fv)
    damped_fv = [Fv[i] + 0.0001 * tf.ones(Fv[i].shape) for i in range(len(Fv))]
    return damped_fv


g_step = tf.train.get_or_create_global_step()

# CG variables
dtype = tf.float32
with tf.name_scope('cg_vars'):
    cg_step = tf.Variable(0, trainable=False, dtype=tf.int32)

    cg_delta = []
    directions = []
    residuals = []
    residual_norm = tf.Variable(0, trainable=False, dtype=tf.float32)
    for w in vars:
        zeros = tf.zeros(w.get_shape(), dtype=dtype)
        delta = tf.Variable(zeros, dtype=dtype, name='delta')
        cg_delta.append(delta)
        d = tf.Variable(zeros, dtype=dtype, name='direction')
        directions.append(d)
        r = tf.Variable(zeros, dtype=dtype, name='residual')
        residuals.append(r)

cg_decay = 0.0
def reset_cg_delta():
    update_delta_0_ops = []
    for delta in cg_delta:
        update_delta = tf.assign(delta, cg_decay * delta)
        update_delta_0_ops.append(update_delta)
    update_delta_0_op = tf.group(*update_delta_0_ops)
    return update_delta_0_op

reset_cg_step = tf.assign(cg_step, 0)
reset_cg_delta_op = reset_cg_delta()

def cg_iter(gradients):
    with tf.name_scope('conjugate_gradient'):
        Ax = Fvp(cg_delta)
        b = [grad for grad in gradients]
        bAx = [Ax - b for b, Ax in zip(b, Ax)]

        condition = tf.equal(cg_step, 0)

        r = [tf.cond(condition, lambda: tf.assign(r,  bax), lambda: r) for r, bax  in zip(residuals, bAx)]
        # r = tf.cond(condition, lambda: tf.assign(residual,  Avar @ delta - bvar), lambda: residual)
        with tf.control_dependencies(r):
            d = [tf.cond(condition, lambda: tf.assign(d, -r), lambda: d) for  d, r in zip(directions, r)]
            # d = tf.cond(condition, lambda: tf.assign(direction, -residual), lambda: direction)

            Ad = Fvp(d)

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
    return tf.less(cg_step, 10)

def cg_body(cg_step, cg_delta, cg_directions, cg_residuals, residual_norm, gradients):
    # with tf.control_dependencies([tf.print(tf.equal(cg_step, 0))]):
    cg_step, cg_delta_update_ops, cg_directions_update_ops, cg_residuals_update_ops, residual_norm = cg_iter(gradients)
    return cg_step, cg_delta_update_ops, cg_directions_update_ops, cg_residuals_update_ops, residual_norm, gradients


with tf.control_dependencies([reset_cg_step, reset_cg_delta_op]):
    gradients = tf.gradients(loss, vars)
    cg_op = tf.while_loop(
        cg_cond,
        cg_body,
        (cg_step, cg_delta, directions, residuals, residual_norm, gradients),
        back_prop=False,
        parallel_iterations=1)

# cg_op_flat = []
# for sublist in cg_op:
#     if isinstance(sublist, list):
#         for item in sublist:
#             cg_op_flat.append(item)

with tf.control_dependencies(cg_op[1]):
    var_update_ops = []
    for i in range(len(vars)):
        # var_update = tf.assign_add(vars[i], -0.1 * gradients[i]) #cg_delta[i])
        var_update = tf.assign_add(vars[i], 0.1 * cg_delta[i])
        var_update_ops.append(var_update)
        # var_update_ops.append(tf.print(cg_delta[i]))
        # var_update_ops.append(tf.print(gradients[i]))

    train_op = tf.group(var_update_ops)

with tf.train.MonitoredTrainingSession() as sess:
    import time
    t1 = time.time()

    while not sess.should_stop():
        stime = time.time()
        global_step_, loss_, accuracy_, _ = sess.run([g_step, loss, accuracy, train_op])
        etime = time.time()
        step_time = etime - stime
        # times.append(step_time)

        t2 = time.time()
        if global_step_ % 1 == 0:
            print ("global_step: %d | loss: %f | accuracy: %s" %
                  (global_step_, loss_, accuracy_))
            print ("FANG time: ", (t2-t1), step_time)


# with tf.Session() as sess:
#     init_op = tf.global_variables_initializer()
#     sess.run(init_op)
#
#     # while True:
#     import time
#     s1 = time.time()
#
#     for i in range(0, x_train.shape[0], bs):
#         x_t, y_t = x_train[i:i+bs], y_train[i:i+bs]
#         # print (x_t.shape, y_t.shape)
#         if x_t.shape[0] < bs:
#             break
#         # import time
#         # s1 = time.time()
#         _, acc, los = sess.run([train_op, accuracy, loss], feed_dict={x: x_t, y: y_t})
#
#         # sess.run(tf.print(cg_step))
#         # cg_out, _, acc, los = sess.run([cg_iter_op, train_op, accuracy, loss], feed_dict={x: x_t, y: y_t})
#         # s2 = time.time()
#         # print ("time: ", s2-s1)
#         print ("Loss: ", los, "; acc: ", acc)
#
#         # time.sleep(0.1)
#     s2 = time.time()
#     print ("time: ", s2-s1)
