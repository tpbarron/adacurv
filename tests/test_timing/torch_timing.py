
# from __future__ import print_function
# import argparse
# import os
# import pickle
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
# import torch.optim as optim
from torch.autograd import Variable
import numpy as onp

# from adacurv.torch.optim.hvp_utils import kl_closure, kl_closure_idx, loss_closure, loss_closure_idx, mean_kl_multinomial
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

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x, return_z=False):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        if return_z:
            return F.log_softmax(x, dim=1), x
        return F.log_softmax(x, dim=1)


def mean_kl_multinomial(new_log_probs, old_log_probs):
    kls = torch.sum(torch.exp(old_log_probs) * (old_log_probs - new_log_probs), dim=1)
    mean_kl = torch.mean(kls)
    return mean_kl

def kl_closure(model, inputs, targets):
    def func():
        new_log_probs = model(inputs)
        old_log_probs = torch.clone(new_log_probs).detach()
        f = mean_kl_multinomial(new_log_probs, old_log_probs)
        return f, list(model.parameters())
    return func

def Fvp(f, x, vector, damping=1e-4):
    vec = Variable(vector, requires_grad=False)
    grad_fo = torch.autograd.grad(f, x, create_graph=True)
    flat_grad = torch.cat([g.contiguous().view(-1) for g in grad_fo])
    h = torch.sum(flat_grad * vec)
    hvp = torch.autograd.grad(h, x, create_graph=True, retain_graph=True)
    hvp_flat = torch.cat([g.contiguous().view(-1) for g in hvp])
    return hvp_flat + damping * vector

def make_fvp_fun(closure):
    """
    Simply create Fvp func that doesn't require theta input so that lanczos and CG can be called
    with generic hvp funcs.
    """
    c, params = closure()
    def f(v):
        hessp = Fvp(c, params, v)
        return hessp.data
    return f


def run():
    # Instantiate model
    model = Net() #.to(device)

    # Load keras MNIST
    (X_train, y_train), (X_test, y_test) = load_mnist()
    x = Variable(torch.from_numpy(X_train[0:128]).float())
    y = Variable(torch.from_numpy(y_train[0:128]).float())

    n_params = 0
    for p in model.parameters():
        n_params += p.numel()

    # Compute Fvp n times where
    # X 1) the first order gradient is computed once and reused.
    # 2) the first order gradient is recomputed for each call

    closure = kl_closure(model, x, y)
    fvp_fn = make_fvp_fun(closure)

    n = 100
    t0 = time.time()
    for i in range(n):
        # print ("i = ", i)
        v = torch.randn(79510)
        fvp_out = fvp_fn(v)
    t1 = time.time()
    print ("PyTorch: Time for " + str(n) + " iters: ", (t1-t0))
