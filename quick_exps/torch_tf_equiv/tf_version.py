import numpy as np

import tensorflow as tf
from adacurv.tf.optimizers import NGDOptimizer, NaturalAdamNGDOptimizer

from keras.datasets import mnist


def load_mnist():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_train = X_train.reshape(-1, 28*28)
    X_test /= 255
    X_test = X_test.reshape(-1, 28*28)
    return (X_train, y_train), (X_test, y_test)


def run(args):
    # set seed
    np.random.seed(0)
    tf.random.set_random_seed(0)

    # generate initial params, set to model
    n_inputs = 784
    n_outputs = 10
    W = np.random.random((n_outputs, n_inputs))
    b = np.random.random((n_outputs,))

    bs = 128

    with tf.name_scope('model'):
        x = tf.placeholder(shape=(bs, 784), dtype='float32', name='input')
        y = tf.placeholder(shape=(bs,), dtype='int64', name='output')

        z = tf.nn.xw_plus_b(x, tf.Variable(W.T.astype(np.float32), dtype=tf.float32), tf.Variable(b.astype(np.float32), tf.float32))
        # h = tf.keras.layers.Flatten()(x)
        # dense_layer = tf.keras.layers.Dense(10,
        #     kernel_initializer=tf.Variable(W, dtype=tf.float32),
        #     bias_initializer=tf.Variable(b, dtype=tf.float32),
        #     activation=None,
        #     use_bias=True)
        # z = dense_layer(x)
        pred = tf.nn.log_softmax(z)

        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=z))
        # loss = -tf.reduce_mean(tf.reduce_sum(tf.one_hot(y, 10) * pred, 1))
        accuracy = tf.reduce_mean(tf.cast(tf.equal(y, tf.argmax(z, axis=1)), dtype=tf.float32))

    g_step = tf.train.get_or_create_global_step()
    common_kwargs = dict(lr=args.lr,
                         curv_type=args.curv_type,
                         betas=(args.beta1, args.beta2),
                         cg_iters=args.cg_iters,
                         cg_residual_tol=args.cg_residual_tol,
                         cg_prev_init_coef=args.cg_prev_init_coef,
                         cg_precondition_empirical=args.cg_precondition_empirical,
                         cg_precondition_regu_coef=args.cg_precondition_regu_coef,
                         cg_precondition_exp=args.cg_precondition_exp,
                         shrinkage_method=args.shrinkage_method,
                         lanczos_amortization=args.lanczos_amortization,
                         lanczos_iters=args.lanczos_iters,
                         batch_size=args.batch_size)
    optimizer = NaturalAdamNGDOptimizer(**common_kwargs)
    train_op = optimizer.minimize(loss, z, global_step=g_step)

    # Load mnist by keras for consistency with tf
    (X_train, y_train), (X_test, y_test) = load_mnist()

    # # compute a few iterations of optimizer and log data
    # for i in range(1):
    #     data = X_train[bs*i:bs*(i+1)]
    #     target = y_train[bs*i:bs*(i+1)]
    #     optimization_step(model, optimizer, data, target)
    #
    # print (optimizer.log.data)

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        print("Trainable vars " + str(len(tf.trainable_variables())) + ": ", sess.run(tf.trainable_variables()))

        for i in range(0, X_train.shape[0], bs):
            x_t, y_t = X_train[i:i+bs], y_train[i:i+bs]

            if x_t.shape[0] < bs:
                break

            step, _, acc, los = sess.run([g_step, train_op, accuracy, loss], feed_dict={x: x_t, y: y_t})

            if i > 1:
                break

            optimizer.log.next_iteration()

    optimizer.log.save_log()
    # print(optimizer.log.data)
