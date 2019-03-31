import tensorflow as tf
import numpy as np
mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

bs = 500

with tf.name_scope('model'):
    x = tf.placeholder(shape=(bs, 28, 28), dtype='float32', name='input')
    y = tf.placeholder(shape=(bs,), dtype='int64', name='output')

    h = tf.keras.layers.Flatten()(x)
    h = tf.keras.layers.Dense(100, activation=tf.nn.relu, use_bias=True)(h)
    z = tf.keras.layers.Dense(10, activation=None, use_bias=True)(h)
    pred = tf.nn.log_softmax(z)
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=z))
    accuracy = tf.reduce_mean(tf.cast(tf.equal(y, tf.argmax(z, axis=1)), dtype=tf.float32))

vars = tf.trainable_variables()
tmp_vars = [tf.Variable(v.initialized_value()) for v in vars]

other_vars = [tf.Variable(tf.random_normal(v.shape)) for v in vars]
grads = tf.gradients(loss, vars)

backup_vars = [tf.assign(tv, v) for tv, v in zip(tmp_vars, vars)]
update_vars = [tf.assign(v, ov) for v, ov in zip(vars, other_vars)]
reset_vars = [tf.assign(v, tv) for v, tv in zip(vars, tmp_vars)]

with tf.Session() as sess:

    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    for i in range(0, x_train.shape[0], bs):
        x_t, y_t = x_train[i:i+bs], y_train[i:i+bs]

        if x_t.shape[0] < bs:
            break

        g_out, acc, los = sess.run([grads, accuracy, loss], feed_dict={x: x_t, y: y_t})
        g_out = [np.sum(np.abs(g)) for g in g_out]
        print ("Grads: ", g_out, "; Loss: ", los, "; acc: ", acc)

        sess.run(backup_vars)
        sess.run(update_vars)

        g_out, acc, los = sess.run([grads, accuracy, loss], feed_dict={x: x_t, y: y_t})
        g_out = [np.sum(np.abs(g)) for g in g_out]
        print ("Grads: ", g_out, "; Loss: ", los, "; acc: ", acc)

        sess.run(reset_vars)
        g_out, acc, los = sess.run([grads, accuracy, loss], feed_dict={x: x_t, y: y_t})
        g_out = [np.sum(np.abs(g)) for g in g_out]
        print ("Grads: ", g_out, "; Loss: ", los, "; acc: ", acc)

        input("")
