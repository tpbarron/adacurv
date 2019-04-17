
# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A basic MNIST example using JAX together with the mini-libraries stax, for
neural network building, and optimizers, for first-order stochastic optimization.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import itertools

import numpy.random as npr

import jax.numpy as np
from jax.config import config
from jax import jit, grad, random
from jax.experimental import optimizers
from jax.experimental import stax
from jax.experimental.stax import Dense, Relu, LogSoftmax
# from examples import datasets
from jax import jacfwd, jacrev



###
# Data
###
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

###
# Jax structure
###

(X_train, y_train), (X_test, y_test) = load_mnist()

def hvp(f, x, v):
    return grad(lambda x: np.vdot(grad(f)(x), v))

def loss(params, batch):
    inputs, targets = batch
    preds = predict(params, inputs)
    return -np.mean(preds * targets)

def accuracy(params, batch):
    inputs, targets = batch
    target_class = np.argmax(targets, axis=1)
    predicted_class = np.argmax(predict(params, inputs), axis=1)
    return np.mean(predicted_class == target_class)

init_random_params, predict = stax.serial(
    Dense(100), Relu,
    Dense(10), LogSoftmax)

def run():
    key = random.PRNGKey(0)

    inputs = X_train[0:128]
    targets = y_train[0:128]

    _, init_params = init_random_params(key, (-1, 28 * 28))

    out = predict(init_params, inputs)
    print ("out: ", out)

    # J = jacfwd(loss)(init_params, (inputs, targets))
    # print("jacfwd result, with shape", J.shape)
    # print(J)

    J = jacrev(loss)(init_params, (inputs, targets))
    print("jacrev result, with shape", J.shape)
    print(J)

    # key, subkey1 = random.split(key, 2)
    # vs = [random.normal(key, size) for size in [(784,100), (100,), (100,10), (10,)]]
    # # # V = random.normal(subkey1, [(784,100), (100), (100,10), (10)])
    # # print (vs)
    # hvp_eval = hvp(loss, init_params, vs)(init_params, (inputs, targets))
