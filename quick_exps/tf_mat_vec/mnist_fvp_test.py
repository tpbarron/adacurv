import numpy as np
import tensorflow as tf

mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# print (x_train.shape, y_train.shape)
print (y_train[0:128].shape)

x = tf.placeholder(shape=(128, 28, 28), dtype='float32', name='input')
x_var = tf.Variable(tf.constant(0.0, shape=(128, 28, 28), dtype='float32'), name='var_input')
x_init = x_var.assign(x)

y = tf.placeholder(shape=(128,10), dtype='float32', name='output')
y_var = tf.Variable(tf.constant(0, shape=(128,10), dtype='float32'), name='var_output')
y_init = y_var.assign(y)

h = tf.layers.Flatten()(x_var)
h = tf.layers.Dense(512, activation=tf.nn.relu)(h)
z = tf.layers.Dense(10, activation=None)(h)
y_pred = tf.nn.softmax(z)
f = tf.reduce_mean(-tf.reduce_sum(y_var * tf.log(y_pred), reduction_indices=[1]))
# y = tf.placeholder(shape=(None,), dtype='int32', name='output')
f = tf.losses.sparse_softmax_cross_entropy(labels=y_var, logits=z)

def hessian_vec_bk(ys, xs, vs, grads=None):
    """Implements Hessian vector product using backward on backward AD.
    Args:
    ys: Loss function.
    xs: Weights, list of tensors.
    vs: List of tensors to multiply, for each weight tensor.
    Returns:
    Hv: Hessian vector product, same size, same shape as xs.
    """
    # Validate the input
    if type(xs) == list:
        if len(vs) != len(xs):
            raise ValueError("xs and vs must have the same length.")
    if grads is None:
        grads = tf.gradients(ys, xs, gate_gradients=True)
    return tf.gradients(grads, xs, vs, gate_gradients=True)

def fisher_vec_bk(ys, xs, vs):
    """Implements Fisher vector product using backward AD.
    Args:
    ys: Loss function, scalar.
    xs: Weights, list of tensors.
    vs: List of tensors to multiply, for each weight tensor.
    Returns:
    J'Jv: Fisher vector product.
    """
    # Validate the input
    if type(xs) == list:
        if len(vs) != len(xs):
            raise ValueError("xs and vs must have the same length.")
    grads = tf.gradients(ys, xs, gate_gradients=True)
    gradsv = list(map(lambda x: tf.reduce_sum(x[0] * x[1]), zip(grads, vs)))
    jv = tf.add_n(gradsv)
    jjv = list(map(lambda x: x * jv, grads))
    return jjv

# from tensorflow_forward_ad import forward_gradients
# def fwd_gradients(ys, xs, d_xs, gate_gradients=True):
#   """Forward-mode pushforward analogous to the pullback defined by tf.gradients.
#   With tf.gradients, grad_ys is the vector being pulled back, and here d_xs is
#   the vector being pushed forward."""
#   v = tf.ones_like(ys) #tf.placeholder(ys[0].dtype, shape=ys[0].get_shape(), name='dummy')  # dummy variable
#   g = tf.gradients(ys, xs, grad_ys=v, gate_gradients=gate_gradients)
#   return tf.gradients(g, v, grad_ys=d_xs, gate_gradients=gate_gradients)
#
# def gauss_newton_vec(ys, zs, xs, vs):
#     """Implements Gauss-Newton vector product.
#     Args:
#     ys: Loss function.
#     zs: Before output layer (input to softmax).
#     xs: Weights, list of tensors.
#     vs: List of perturbation vector for each weight tensor.
#     Returns:
#     J'HJv: Guass-Newton vector product.
#     """
#     # Validate the input
#     if type(xs) == list:
#         if len(vs) != len(xs):
#             raise ValueError("xs and vs must have the same length.")
#
#     grads_z = tf.gradients(ys, zs, gate_gradients=True)
#     print_grad_z = tf.print(grads_z[0].shape)
#     hjv = fwd_gradients(grads_z, xs, vs, gate_gradients=True)
#     print_hjv = tf.print(hjv[0].shape)
#     # jhjv = tf.gradients(zs, xs, hjv, gate_gradients=True)
#     return None, print_grad_z,  print_hjv

# from tensorflow_forward_ad.second_order import fisher_vec_fw


shapes = [(784, 512), (512,), (512, 10), (10,)]
v1 = tf.random.uniform(shape=shapes[0], dtype='float32', name='vector') #tf.placeholder(shape=shapes[0], dtype='float32', name='vector')
v2 = tf.random.uniform(shape=shapes[1], dtype='float32', name='vector') #tf.placeholder(shape=shapes[1], dtype='float32', name='vector')
v3 = tf.random.uniform(shape=shapes[2], dtype='float32', name='vector') #tf.placeholder(shape=shapes[2], dtype='float32', name='vector')
v4 = tf.random.uniform(shape=shapes[3], dtype='float32', name='vector') #tf.placeholder(shape=shapes[3], dtype='float32', name='vector')
vectors = [v1, v2, v3, v4]
vars = tf.trainable_variables()[2:]
print (vars)

# Fvp = hessian_vec_bk(f, vars, vectors)
Fvp = fisher_vec_bk(f, vars, vectors)
# res = tf.Variable(tf.constant(0.0, shape=(10,1)), name='fvp_result')
# print (Fvp)
# Fvp_with_assign = res.assign(tf.reshape(Fvp, shape=(10,1)))

Fvp_blocks = [fisher_vec_bk(f, [vars[i]], [vectors[i]]) for i in range(len(vectors))] #[v1, v2, v3, v4])
# GNvp, print_grad_z, print_hjv = gauss_newton_vec(f, z, [tf.trainable_variables()[0]], [v1])

def indices_to_one_hot(data, nb_classes):
    """Convert an iterable of indices to one-hot encoded labels."""
    targets = np.array(data).reshape(-1)
    return np.eye(nb_classes)[targets]

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    sess.run([x_init, y_init], feed_dict={x:x_train[0:128], y:indices_to_one_hot(y_train[0:128], 10)})
    # out = sess.run(f, feed_dict={x:x_train[0:128], y:y_train[0:128][:,np.newaxis]})

    import time
    n = 10
    s = time.time()
    for i in range(n):
        fvp = sess.run(Fvp)
    e = time.time()
    print ("Fvp avg time: ", (e-s)/n)

    s = time.time()
    for i in range(n):
        fvp = sess.run(Fvp_blocks)
    e = time.time()
    print ("Fvp avg time: ", (e-s)/n)

    # s = time.time()
    # for i in range(n):
    #     fvp = sess.run(Fvp_with_assign) #, feed_dict={x:x_train[0:128],
    #                                    # y:y_train[0:128]}) #,
    #                                    # v1:np.random.random(shapes[0]),
    #                                    # v2:np.random.random(shapes[1]),
    #                                    # v3:np.random.random(shapes[2]),
    #                                    # v4:np.random.random(shapes[3])})
    #     # print ("fvp: ", fvp)
    # e = time.time()
    # print ("Fvp assign avg time: ", (e-s)/n)

    # s = time.time()
    # for i in range(n):
    #     fvp_blocks = sess.run(Fvp_blocks, feed_dict={x:x_train[0:128],
    #                                    y:y_train[0:128]})
    #                                    # v1:np.random.random(shapes[0]),
    #                                    # v2:np.random.random(shapes[1]),
    #                                    # v3:np.random.random(shapes[2]),
    #                                    # v4:np.random.random(shapes[3])})
    #     # print ("fvp: ", fvp)
    # e = time.time()
    # print ("Fvp blocks avg time: ", (e-s)/n)

    # sess.run((print_grad_z, print_hjv), feed_dict={x:x_train[0:128],
    #                                y:y_train[0:128][:,np.newaxis],
    #                                v1:np.random.random(shapes[0])})
    # gnvp = sess.run(Gnvp, feed_dict={x:x_train[0:128],
    #                                  y:y_train[0:128][:,np.newaxis],
    #                                  v1:np.random.random(shapes[0])})
    # print ("gnvp: ", gnvp)
