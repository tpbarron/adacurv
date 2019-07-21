
import jax.numpy as np

from jax import jit, grad, jvp
from jax import random
from jax.flatten_util import ravel_pytree

from torchvision import datasets, transforms

import numpy_loader
import optimizers

mnist_train_dataset = datasets.MNIST('/tmp/mnist/', train=True, download=True,
               transform=transforms.Compose([
                   transforms.ToTensor(),
                   transforms.Normalize((0.1307,), (0.3081,)),
                   numpy_loader.CastToNumpy(),
               ]))

mnist_test_dataset = datasets.MNIST('/tmp/mnist/', train=False, download=True,
               transform=transforms.Compose([
                   transforms.ToTensor(),
                   transforms.Normalize((0.1307,), (0.3081,)),
                   numpy_loader.CastToNumpy(),
               ]))

# TODO(trevor, nikhil): The use of NumpyLoader could be made more efficient by loading to PyTorch
# tensors with multiple workers and then casting to numpy.
training_generator = numpy_loader.NumpyLoader(mnist_train_dataset, batch_size=250, num_workers=0)
testing_generator = numpy_loader.NumpyLoader(mnist_test_dataset, batch_size=1000, num_workers=0)

###
# Model
###

from jax.experimental import stax
from jax.experimental.stax import Dense, Relu, Tanh, LogSoftmax

# TODO: this implementation seems to be more sensitive to hyperparams then the pytorch one. Maybe
# check weight init.

# Use stax to set up network initialization and evaluation functions
net_init, net_apply = stax.serial(
    Dense(500), Tanh,
    Dense(10), LogSoftmax,
)

# Initialize parameters, not committing to a batch shape
rng = random.PRNGKey(0)
in_shape = (-1, 784)
out_shape, net_params = net_init(rng, in_shape)

###
# Loss calculation, negative log likelihood
###

def loss(params, batch):
  inputs, targets = batch
  preds = net_apply(params, inputs)
  return -np.mean(preds * targets)

lr = 0.00025
# Use optimizers to set optimizer initialization and update functions
opt_init, opt_update, get_params = optimizers.sgd(1.0) #optimizers.exponential_decay(lr, 1000, 0.95))

###
# Update step
###

from adacurv.jax.utils.cg import cg_solve_jax_hvp

def hvp(loss, params, batch, v):
  """Computes the hessian vector product Hv.
  This implementation uses forward-over-reverse mode for computing the hvp.
  Args:
    loss: function computing the loss with signature
      loss(params, batch).
    params: pytree for the parameters of the model.
    batch:  A batch of data. Any format is fine as long as it is a valid input
      to loss(params, batch).
    v: pytree of the same structure as params.
  Returns:
    hvp: array of shape [num_params] equal to Hv where H is the hessian.
  """
  loss_fn = lambda x: loss(x, batch)
  return jvp(grad(loss_fn), [params], [v])[1]


def natural_grad(loss):
    """
    Returns a function that computes the natural gradient by truncated newton given params, batch.
    """
    def ng_fun(params, batch):
        g = grad(loss)(params, batch)
        flat_g, unravel = ravel_pytree(g)

        hvp_fn = lambda v: ravel_pytree(hvp(loss, params, batch, unravel(v)))[0]
        ng = unravel(cg_solve_jax_hvp(hvp_fn,
                                      flat_g,
                                      x_0=np.array(flat_g, copy=True),
                                      cg_iters=10))
        return ng, g

    return ng_fun

@jit
def step(i, opt_state, batch):
  params = get_params(opt_state)
  ng, g = natural_grad(loss)(params, batch)

  #TODO(trevor, nikhil): find a way to move step size computation to the optimizer. Right now the
  # LR is 1.0 and this scales the gradient.
  # step_i = lr
  step_i = lr * 0.95 ** (i / 50.0)
  alpha = np.sqrt(np.abs(step_i / (np.dot(ravel_pytree(g)[0], ravel_pytree(ng)[0]) + 1e-20)))
  # print ("Using learning rate: ", i,  alpha)
  ng_flat, unravel = ravel_pytree(ng)
  ng_contract = unravel(ng_flat * alpha)

  return opt_update(i, ng_contract, opt_state)

###
# Training loop
###

def accuracy(params, batch):
    inputs, targets = batch
    target_class = np.argmax(targets, axis=-1)
    predicted_class = np.argmax(net_apply(params, inputs), axis=-1)
    return np.sum(predicted_class == target_class)

# TODO(trevor, nikhil): see if we can replace the onehot, just pass integer classes to log likelihood
# loss like pytorch does.
def one_hot(x, k, dtype=np.float32):
  """Create a one-hot encoding of x of size k."""
  return np.array(x[:, None] == np.arange(k), dtype)

n_targets = 10

def test(opt_state):
    total_correct = 0
    for i, batch in enumerate(testing_generator):
        x, y = batch
        x = x.reshape(-1, 784)
        y = one_hot(y, n_targets)
        total_correct += accuracy(get_params(opt_state), (x, y))

    print ("Test accuracy: ", total_correct / 10000.0 * 100.0)

def train():
    # Optimize parameters in a loop
    opt_state = opt_init(net_params)
    test(opt_state)

    for epoch in range(10):
        print ("Epoch ", epoch)
        for i, batch in enumerate(training_generator):
            if i % 10 == 0:
                print (" > batch: ", i)
            x, y = batch
            x = x.reshape(-1, 784)
            y = one_hot(y, n_targets)
            opt_state = step(i, opt_state, (x, y))


        test(opt_state)

if __name__ == "__main__":
    train()
