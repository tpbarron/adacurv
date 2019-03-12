import numpy as np
import tensorflow as tf
import fashion_mnist_data as mnist

# Inverse update ops will be run every _INVERT_EVRY iterations.
_INVERT_EVERY = 10

# Covariance matrices will be update  _COV_UPDATE_EVERY iterations.
_COV_UPDATE_EVERY = 1

# Displays loss every _REPORT_EVERY iterations.
_REPORT_EVERY = 1

# Use manual registration
_USE_MANUAL_REG = False

def fc_layer(layer_id, inputs, output_size):
  """Builds a fully connected layer.

  Args:
    layer_id: int. Integer ID for this layer's variables.
    inputs: Tensor of shape [num_examples, input_size]. Each row corresponds
      to a single example.
    output_size: int. Number of output dimensions after fully connected layer.

  Returns:
    preactivations: Tensor of shape [num_examples, output_size]. Values of the
      layer immediately before the activation function.
    activations: Tensor of shape [num_examples, output_size]. Values of the
      layer immediately after the activation function.
    params: Tuple of (weights, bias), parameters for this layer.
  """
  # TODO(b/67004004): Delete this function and rely on tf.layers exclusively.
  layer = tf.layers.Dense(
      output_size,
      kernel_initializer=tf.random_normal_initializer(),
      name="fc_%d" % layer_id)
  preactivations = layer(inputs)
  activations = tf.nn.relu(preactivations)

  # layer.weights is a list. This converts it a (hashable) tuple.
  return preactivations, activations, (layer.kernel, layer.bias)


def conv_layer(layer_id, inputs, kernel_size, out_channels):
  """Builds a convolutional layer with ReLU non-linearity.

  Args:
    layer_id: int. Integer ID for this layer's variables.
    inputs: Tensor of shape [num_examples, width, height, in_channels]. Each row
      corresponds to a single example.
    kernel_size: int. Width and height of the convolution kernel. The kernel is
      assumed to be square.
    out_channels: int. Number of output features per pixel.

  Returns:
    preactivations: Tensor of shape [num_examples, width, height, out_channels].
      Values of the layer immediately before the activation function.
    activations: Tensor of shape [num_examples, width, height, out_channels].
      Values of the layer immediately after the activation function.
    params: Tuple of (kernel, bias), parameters for this layer.
  """
  # TODO(b/67004004): Delete this function and rely on tf.layers exclusively.
  layer = tf.layers.Conv2D(
      out_channels,
      kernel_size=[kernel_size, kernel_size],
      kernel_initializer=tf.random_normal_initializer(stddev=0.01),
      padding="SAME",
      name="conv_%d" % layer_id)
  preactivations = layer(inputs)
  activations = tf.nn.relu(preactivations)

  # layer.weights is a list. This converts it a (hashable) tuple.
  return preactivations, activations, (layer.kernel, layer.bias)


def max_pool_layer(layer_id, inputs, kernel_size, stride):
  """Build a max-pooling layer.

  Args:
    layer_id: int. Integer ID for this layer's variables.
    inputs: Tensor of shape [num_examples, width, height, in_channels]. Each row
      corresponds to a single example.
    kernel_size: int. Width and height to pool over per input channel. The
      kernel is assumed to be square.
    stride: int. Step size between pooling operations.

  Returns:
    Tensor of shape [num_examples, width/stride, height/stride, out_channels].
    Result of applying max pooling to 'inputs'.
  """
  # TODO(b/67004004): Delete this function and rely on tf.layers exclusively.
  with tf.variable_scope("pool_%d" % layer_id):
    return tf.nn.max_pool(
        inputs, [1, kernel_size, kernel_size, 1], [1, stride, stride, 1],
        padding="SAME",
        name="pool")


def build_model(examples,
                labels,
                num_labels): #
                # layer_collection,
                # register_layers_manually=False):
  """Builds a ConvNet classification model.

  Args:
    examples: Tensor of shape [num_examples, num_features]. Represents inputs of
      model.
    labels: Tensor of shape [num_examples]. Contains integer IDs to be predicted
      by softmax for each example.
    num_labels: int. Number of distinct values 'labels' can take on.
    layer_collection: LayerCollection instance. Layers will be registered here.
    register_layers_manually: bool. If True then register the layers with
      layer_collection manually. (Default: False)

  Returns:
    loss: 0-D Tensor representing loss to be minimized.
    accuracy: 0-D Tensor representing model's accuracy.
  """
  # Build a ConvNet. For each layer with parameters, we'll keep track of the
  # preactivations, activations, weights, and bias.
  tf.logging.info("Building model.")
  # pre0, act0, params0 = conv_layer(
  #     layer_id=0, inputs=examples, kernel_size=5, out_channels=16)
  # act1 = max_pool_layer(layer_id=1, inputs=act0, kernel_size=3, stride=2)
  # pre2, act2, params2 = conv_layer(
  #     layer_id=2, inputs=act1, kernel_size=5, out_channels=32)
  # act3 = max_pool_layer(layer_id=3, inputs=act2, kernel_size=3, stride=2)
  # flat_act3 = tf.reshape(act3, shape=[-1, int(np.prod(act3.shape[1:4]))])
  # pre4, act4, params4 = fc_layer(
  #     layer_id=4, inputs=flat_act3, output_size=256)
  pre4, act4, params4 = fc_layer(
      layer_id=4, inputs=examples, output_size=256)
  logits, _, params5 = fc_layer(
      layer_id=4, inputs=act4, output_size=num_labels)

  loss = tf.reduce_mean(
      tf.nn.sparse_softmax_cross_entropy_with_logits(
          labels=labels, logits=logits))
  accuracy = tf.reduce_mean(
      tf.cast(tf.equal(labels, tf.argmax(logits, axis=1)), dtype=tf.float32))

  with tf.device("/cpu:0"):
    tf.summary.scalar("loss", loss)
    tf.summary.scalar("accuracy", accuracy)

  # layer_collection.register_softmax_cross_entropy_loss(
  #     logits, name="logits")
  #
  # if register_layers_manually:
  #   layer_collection.register_conv2d(params0, (1, 1, 1, 1), "SAME", examples,
  #                                    pre0)
  #   layer_collection.register_conv2d(params2, (1, 1, 1, 1), "SAME", act1,
  #                                    pre2)
  #   layer_collection.register_fully_connected(params4, flat_act3, logits)

  return loss, accuracy


def minimize_loss_single_machine(loss,
                                 accuracy):
                                 # layer_collection,
                                 # device=None,
                                 # session_config=None):
  """Minimize loss with K-FAC on a single machine.

  Creates `PeriodicInvCovUpdateKfacOpt` which handles inverse and covariance
  computation op placement and execution. A single Session is responsible for
  running all of K-FAC's ops. The covariance and inverse update ops are placed
  on `device`. All model variables are on CPU.

  Args:
    loss: 0-D Tensor. Loss to be minimized.
    accuracy: 0-D Tensor. Accuracy of classifier on current minibatch.
    layer_collection: LayerCollection instance describing model architecture.
      Used by K-FAC to construct preconditioner.
    device: string or None. The covariance and inverse update ops are run on
      this device. If empty or None, the default device will be used.
      (Default: None)
    session_config: None or tf.ConfigProto. Configuration for tf.Session().

  Returns:
    final value for 'accuracy'.
  """
  from adacurv.tf.optimizers import NGDOptimizer
  g_step = tf.train.get_or_create_global_step()
  # optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
  optimizer = NGDOptimizer(learning_rate=0.1)

  train_op = optimizer.minimize(loss, global_step=g_step)

  tf.logging.info("Starting training.")
  with tf.train.MonitoredTrainingSession() as sess:
    while not sess.should_stop():
      global_step_, loss_, accuracy_, _ = sess.run(
        [g_step, loss, accuracy, train_op])

      if global_step_ % _REPORT_EVERY == 0:
        print ("global_step: %d | loss: %f | accuracy: %s" %
                    (global_step_, loss_, accuracy_))
        tf.logging.info("global_step: %d | loss: %f | accuracy: %s",
                    global_step_, loss_, accuracy_)

  return accuracy_


def train_mnist_single_machine(num_epochs,
                               use_fake_data=False,
                               device=None,
                               manual_op_exec=False):
  """Train a ConvNet on MNIST.
  Args:
    num_epochs: int. Number of passes to make over the training set.
    use_fake_data: bool. If True, generate a synthetic dataset.
    device: string or None. The covariance and inverse update ops are run on
      this device. If empty or None, the default device will be used.
      (Default: None)
    manual_op_exec: bool, If `True` then `minimize_loss_single_machine_manual`
      is called for training which handles inverse and covariance computation.
      This is shown only for illustrative purpose. Otherwise
      `minimize_loss_single_machine` is called which relies on
      `PeriodicInvCovUpdateOpt` for op placement and execution.
  Returns:
    accuracy of model on the final minibatch of training data.
  """
  # Load a dataset.
  print ("Loading MNIST into memory.")
  tf.logging.info("Loading MNIST into memory.")
  (examples, labels) = mnist.load_mnist_as_iterator(num_epochs,
                                                    250,
                                                    use_fake_data=use_fake_data,
                                                    flatten_images=True)

  # Build a ConvNet.
  loss, accuracy = build_model(
      examples, labels, num_labels=10)

  return minimize_loss_single_machine(
        loss, accuracy) #, device=device, session_config=config)


if __name__ == "__main__":
    train_mnist_single_machine(1)
