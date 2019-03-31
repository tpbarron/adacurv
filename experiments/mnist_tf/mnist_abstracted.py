import numpy as np
import tensorflow as tf

from adacurv.tf.optimizers import NGDOptimizer, NaturalAdamNGDOptimizer, NaturalAdagradNGDOptimizer

mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

bs = 250

with tf.name_scope('model'):
    x = tf.placeholder(shape=(bs, 28, 28), dtype='float32', name='input')
    y = tf.placeholder(shape=(bs,), dtype='int64', name='output')

    h = tf.keras.layers.Flatten()(x)
    # z = tf.keras.layers.Dense(10, activation=None, use_bias=False)(h)
    h = tf.keras.layers.Dense(100, activation=tf.nn.relu, use_bias=True)(h)
    z = tf.keras.layers.Dense(10, activation=None, use_bias=True)(h)
    pred = tf.nn.log_softmax(z)
    # kl = tf.reduce_sum(tf.exp(pred) * (pred - pred))

    # loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=z))
    loss = -tf.reduce_mean(tf.reduce_sum(tf.one_hot(y, 10) * pred, 1))
    accuracy = tf.reduce_mean(tf.cast(tf.equal(y, tf.argmax(z, axis=1)), dtype=tf.float32))

tf.summary.scalar('loss', loss)
tf.summary.scalar('accuracy', accuracy)

vars = tf.trainable_variables()
g_step = tf.train.get_or_create_global_step()

algo = 'adam_ngd'
cg_decay = 0.0
betas = (0.1, 0.1)
learn_rate = tf.placeholder(tf.float32, shape=[])


if algo == 'adam':
    train_op = tf.train.AdamOptimizer(learning_rate=learn_rate).minimize(loss, global_step=g_step)
elif algo == 'adam_ngd':
    # train_op = NaturalAdagradNGDOptimizer(learning_rate=learn_rate,
    #                                       cg_decay=cg_decay).minimize(loss,
    #                                                                   z,
    #                                                                   global_step=g_step)
    train_op = NaturalAdamNGDOptimizer(learning_rate=learn_rate,
                                       cg_decay=cg_decay,
                                       beta1=betas[0],
                                       beta2=betas[1]).minimize(loss, z, global_step=g_step)
elif algo == 'ngd':
    train_op = NGDOptimizer(cg_decay=cg_decay).minimize(loss, z, global_step=g_step)

merged = tf.summary.merge_all()

def compute_decayed_lr(epoch):
    init_lr = 0.0005
    decayed_lr = init_lr
    for i in range(epoch):
        decayed_lr *= 1/np.sqrt(i+1)
    print ("decayed_lr: ", decayed_lr)
    return decayed_lr

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    file_str = algo+'3_decay'+str(cg_decay)+'_betas'+str(betas[0])+'_'+str(betas[1])+'_'+str(bs)
    train_writer = tf.summary.FileWriter('results/train/'+file_str, sess.graph)

    iter = 0
    for ep in range(5):
        for i in range(0, x_train.shape[0], bs):
            x_t, y_t = x_train[i:i+bs], y_train[i:i+bs]

            if x_t.shape[0] < bs:
                break

            summary, step, _, acc, los = sess.run([merged, g_step, train_op, accuracy, loss], feed_dict={x: x_t, y: y_t, learn_rate: compute_decayed_lr(ep+1)})
            train_writer.add_summary(summary, iter)
            iter += 1

            print ("Global step: ", step, "; Loss: ", los, "; acc: ", acc)

            # if iter > 5:
            #     exit()
