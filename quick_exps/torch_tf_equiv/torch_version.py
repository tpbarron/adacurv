import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

import numpy as np

from keras.datasets import mnist
from adacurv.torch.optim.hvp_utils import kl_closure, mean_kl_multinomial

class Model(nn.Module):

    def __init__(self, n_inputs, n_outputs, param_inits=[]):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(n_inputs, n_outputs)

        # print (type(self.fc1.weight))
        # input("")
        self.fc1.weight.data = torch.from_numpy(param_inits[0]).float()
        self.fc1.bias.data = torch.from_numpy(param_inits[1]).float()

    def forward(self, x):
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)

def load_mnist():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_train = X_train.reshape(-1, 28*28)
    X_test /= 255
    X_test = X_test.reshape(-1, 28*28)
    return (X_train, y_train), (X_test, y_test)

def optimization_step(model, optimizer, data, target):
    model.train()
    data = Variable(torch.from_numpy(data).float())
    target = Variable(torch.from_numpy(target).long())
    optimizer.zero_grad()
    output = model(data)
    loss = F.nll_loss(output, target)
    loss.backward()

    closure = kl_closure(model, data, target, mean_kl_multinomial)
    optimizer.step(closure)


def run(args):
    # set seed
    np.random.seed(0)
    torch.manual_seed(0)

    # generate initial params, set to model
    n_inputs = 784
    n_outputs = 10
    W = np.random.random((n_outputs, n_inputs))
    b = np.random.random((n_outputs,))

    # print ("W, b: ", W, b)
    # input("")
    model = Model(n_inputs, n_outputs, param_inits=[W, b])

    # Load mnist by keras for consistency with tf
    (X_train, y_train), (X_test, y_test) = load_mnist()

    # fix batch size
    bs = 128

    # create optimizer
    import adacurv.torch.optim as fisher_optim
    common_kwargs = dict(lr=args.lr,
                         curv_type=args.curv_type,
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

    optimizer = fisher_optim.NaturalAdam(model.parameters(),
                                         **common_kwargs,
                                         betas=(args.beta1, args.beta2),
                                         assume_locally_linear=args.approx_adaptive)

    # compute a few iterations of optimizer and log data
    for i in range(1):
        data = X_train[bs*i:bs*(i+1)]
        target = y_train[bs*i:bs*(i+1)]
        optimization_step(model, optimizer, data, target)

    # print (optimizer.log.data)
