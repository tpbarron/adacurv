import jax.numpy as np
from jax import random, jacrev, jacfwd, vjp, jvp, linearize, jit
from jax.experimental import stax
from jax.experimental.stax import Conv, Dense, MaxPool, Relu, Flatten, LogSoftmax

from functools import partial

# Use stax to set up network initialization and evaluation functions
net_init, net_apply = stax.serial(
    Dense(64), Relu,
    Dense(10), LogSoftmax,
)

# Initialize parameters, not committing to a batch shape
rng = random.PRNGKey(0)
in_shape = (-1, 32)
out_shape, net_params = net_init(rng, in_shape)

# Apply network to dummy inputs
inputs = np.zeros((1, 32))
# predictions = net_apply(net_params, inputs)
# print ("pred: ", predictions)

def net_apply_reverse(inputs, net_params):
    return net_apply(net_params, inputs)

@jit
def test_loss(net_params, inputs):
    return np.sum(net_apply(net_params, inputs))

primals_out, vjpfun = vjp(partial(net_apply_reverse, inputs), net_params)
print (primals_out)

primals_out, jvpfun = linearize(partial(net_apply_reverse, inputs), net_params)
# primals_out, vp = jvp(net_apply, (net_params, inputs), random.normal(rng, (1, 256)))
print (primals_out)
input("")

for i in range(10):
    import time
    s = time.time()
    out = vjpfun(random.normal(rng, (1, 10)))
    e = time.time()
    print ("vjp time: ", (e - s))

    s = time.time()
    out = jvpfun(net_params)
    # print (out)
    e = time.time()
    print ("jvp time: ", (e - s))

# for i in range(10):
#     inputs = random.normal(rng, (10, 256))
#     import time
#     s = time.time()
#     jac = jacrev(test_loss, argnums=0)(net_params, inputs)
#     e = time.time()
#     print ("Jax rev time: ", (e - s))
#
#     s = time.time()
#     jac = jacfwd(test_loss, argnums=0)(net_params, inputs)
#     e = time.time()
#     print ("Jax fwd time: ", (e - s))
