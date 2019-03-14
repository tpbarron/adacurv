import tensorflow as tf
from tensorflow_forward_ad import forward_gradients

# Automatic differentiation.
x = tf.constant(1.0)
y = tf.square(x)
dydx = forward_gradients(y, x)
sess = tf.Session()
print(sess.run(dydx))  # [2.0].

# Computes Jacobian-vector product.
x = tf.ones([5, 10])
y = tf.square(x)
v = tf.ones([5, 10]) * 2
Jv = forward_gradients(y, x, v)
sess = tf.Session()
print(sess.run(Jv))  # [array([[ 4.,  4.,  4.,  4., ...
