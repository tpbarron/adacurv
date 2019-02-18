import copy
import numpy as np
import torch
from torch.autograd import Variable
import torch.autograd as autograd
from torch.nn.utils import vector_to_parameters, parameters_to_vector
from fisher.utils.convert_gradients import gradients_to_vector, vector_to_gradients

###
# KL functions for use with the natural gradient variants.
###

def mean_kl_multinomial(new_log_probs, old_log_probs):
    kls = torch.sum(torch.exp(old_log_probs) * (old_log_probs - new_log_probs), dim=1)
    mean_kl = torch.mean(kls)
    return mean_kl

def kl_gaussian(new_log_probs, old_log_probs):
    raise NotImplementedError

###
# Loss closures
###

def kl_closure(model, inputs, targets, kl_fn):
    def func(params):
        old_params = parameters_to_vector(model.parameters())
        if isinstance(params, Variable):
            vector_to_parameters(params, model.parameters())
        else:
            vector_to_parameters(parameters_to_vector(params), model.parameters())
        new_log_probs = model(inputs)
        old_log_probs = torch.clone(new_log_probs).detach()
        f = kl_fn(new_log_probs, old_log_probs)
        tmp_params = list(model.parameters())
        vector_to_parameters(old_params, model.parameters())
        return f, tmp_params
    return func

def kl_closure_idx(model, inputs, targets, kl_fn):
    def func(params, params_i, params_j):
        # print ("Params inpute: ", type(params))
        import time

        old_params = list(model.parameters()) #parameters_to_vector(model.parameters())

        t1s = time.time()
        cur_params = [v.clone() for v in old_params]
        cur_params[params_i:params_j] = params
        t1e = time.time()
        # print ("klidx 0: ", (t1e-t1s))

        t1s = time.time()
        vector_to_parameters(parameters_to_vector(cur_params), model.parameters())
        t1e = time.time()
        # print ("klidx 1: ", (t1e-t1s))

        new_log_probs = model(inputs)
        old_log_probs = torch.clone(new_log_probs).detach()
        f = kl_fn(new_log_probs, old_log_probs)

        t1s = time.time()
        tmp_params = list(model.parameters())[params_i:params_j]
        vector_to_parameters(parameters_to_vector(old_params), model.parameters())
        t1e = time.time()
        # print ("klidx 2: ", (t1e-t1s))

        return f, tmp_params
    return func

def loss_closure(model, inputs, targets, loss_fn):
    def func(params):
        old_params = parameters_to_vector(model.parameters())
        if isinstance(params, Variable):
            vector_to_parameters(params, model.parameters())
        else:
            vector_to_parameters(parameters_to_vector(params), model.parameters())
        outputs, z = model.forward(inputs, return_z=True)
        f = loss_fn(outputs, targets)
        tmp_params = list(model.parameters())
        vector_to_parameters(old_params, model.parameters())
        return f, z, tmp_params
    return func

def loss_closure_idx(model, inputs, targets, loss_fn):
    """
    Use me for block diagonal optimization.
    """
    def func(params, params_i, params_j):
        old_params = list(model.parameters()) #parameters_to_vector(model.parameters())

        cur_params = [v.clone() for v in old_params]
        cur_params[params_i:params_j] = params
        vector_to_parameters(parameters_to_vector(cur_params), model.parameters())

        outputs, z = model(inputs, return_z=True)
        f = loss_fn(outputs, targets)

        tmp_params = list(model.parameters())[params_i:params_j]
        vector_to_parameters(parameters_to_vector(old_params), model.parameters())

        return f, z, tmp_params
    return func

###
# R-op implemented as two L-ops
# See: https://j-towns.github.io/2017/06/12/A-new-trick.html
# Does not seem to be more efficient since PyTorch cannot know the graph structure ahead of time
# but may be more intuitive. Only used in Gauss-Newton vector product.
###

def Rop(ys, xs, vs):
    """
    ys: outputs of graph
    xs: variable to take derivative w.r.t.; usually a set of parameters in PyTorch
    vs: the vector compute jvp for.
    """
    ws = ys.clone().detach().requires_grad_(True)
    gs = torch.autograd.grad(ys, xs, grad_outputs=ws, create_graph=True, retain_graph=True, allow_unused=False)
    gs = torch.cat([g.contiguous().view(-1) for g in gs])
    re = torch.autograd.grad(gs, ws, grad_outputs=vs, create_graph=True, retain_graph=True, allow_unused=False)
    return re

###
# Fvp function by double backprop
###

def Fvp(f, x, vector, damping=1e-4):
    vec = Variable(vector, requires_grad=False)
    grad_fo = torch.autograd.grad(f, x, create_graph=True)
    flat_grad = torch.cat([g.contiguous().view(-1) for g in grad_fo])
    h = torch.sum(flat_grad * vec)
    hvp = torch.autograd.grad(h, x, create_graph=True, retain_graph=True)
    hvp_flat = torch.cat([g.contiguous().view(-1) for g in hvp])
    return hvp_flat + damping * vector


###
# Hessian vector product - this is the same as Fvp but f should be the loss not the KL / log-likelihood
###

def Hvp(f, x, vector, damping=1e-4):
    vec = Variable(vector, requires_grad=False)
    grad_fo = torch.autograd.grad(f, x, create_graph=True)
    flat_grad = torch.cat([g.contiguous().view(-1) for g in grad_fo])
    h = torch.sum(flat_grad * vec)
    hvp = torch.autograd.grad(h, x, create_graph=True, retain_graph=True)
    hvp_flat = torch.cat([g.contiguous().view(-1) for g in hvp])
    return hvp_flat + damping * vector


###
# Gauss-Newton vector product
###

def GNvp(f, z, x, v):
    """
    f: loss
    z: pre-nonlinearity output of last layer
    x: parameters to differentiate w.r.t.
    v: vector to compute Gv
    """
    vec = Variable(v, requires_grad=False)
    grads_z = torch.autograd.grad(f, z, create_graph=True, retain_graph=True)[0]
    hjv = Rop(grads_z, x, vec)
    jhjv = torch.autograd.grad(z, x, grad_outputs=hjv, create_graph=True, retain_graph=True)
    jhjv_flat = torch.cat([g.contiguous().view(-1) for g in jhjv])
    return jhjv_flat

###
# Build true Fisher
###

def F(model, inputs, outputs, kl_fn, damping=1e-4):
    new_log_probs = model(inputs)
    old_log_probs = torch.clone(new_log_probs).detach()

    mean_kl = kl_fn(new_log_probs, old_log_probs)
    loss_grad = autograd.grad(mean_kl, model.parameters(), create_graph=True)
    cnt = 0
    for g in loss_grad:
        g_vector = g.contiguous().view(-1) if cnt == 0 else torch.cat([g_vector, g.contiguous().view(-1)])
        cnt = 1
    l = g_vector.size(0)
    hessian = torch.zeros(l, l)
    for idx in range(l):
        grad2rd = autograd.grad(g_vector[idx], model.parameters(), create_graph=True)
        cnt = 0
        for g in grad2rd:
            g2 = g.contiguous().view(-1) if cnt == 0 else torch.cat([g2, g.contiguous().view(-1)])
            cnt = 1
        hessian[idx] = g2
    return hessian.cpu().data #.numpy()

def build_F(model, inputs, outputs, kl_fn, regu_coef=0.0):
    def Fvp_fn(theta):
        # import time
        # s = time.time()
        # theta should be a parameter vector.
        temp_model = copy.deepcopy(model)
        vector_to_parameters(theta, temp_model.parameters())
        full_inp = [temp_model, inputs, outputs, kl_fn, regu_coef]
        H = eval_F(*full_inp)
        return H
    return Fvp_fn
