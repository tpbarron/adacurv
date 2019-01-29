import copy
import numpy as np
import torch
from torch.autograd import Variable
import torch.autograd as autograd
from torch.nn.utils import vector_to_parameters, parameters_to_vector
from fisher.utils.convert_gradients import gradients_to_vector, vector_to_gradients

###
# KL functions
###

def mean_kl_multinomial(new_log_probs, old_log_probs):
    kls = torch.sum(torch.exp(old_log_probs) * (old_log_probs - new_log_probs), dim=1)
    mean_kl = torch.mean(kls)
    return mean_kl

def kl_gaussian():
    pass

###
# Fvp function by double backprop
###

def Fvp(model, inputs, outputs, kl_fn, vector, damping=1e-4):
    vec = Variable(vector, requires_grad=False)

    new_log_probs = model(inputs)
    old_log_probs = torch.clone(new_log_probs).detach()
    mean_kl = kl_fn(new_log_probs, old_log_probs)

    grad_fo = torch.autograd.grad(mean_kl, model.parameters(), create_graph=True)
    flat_grad = torch.cat([g.contiguous().view(-1) for g in grad_fo])
    h = torch.sum(flat_grad * vec)
    hvp = torch.autograd.grad(h, model.parameters(), create_graph=True, retain_graph=True)
    hvp_flat = torch.cat([g.contiguous().view(-1) for g in hvp])

    return hvp_flat + damping * vector

def build_Fvp(model, inputs, outputs, kl_fn, regu_coef=0.0):
    def Fvp_fn(theta, v, return_model=False):
        # import time
        # s = time.time()
        # theta should be a parameter vector.
        temp_model = copy.deepcopy(model)
        vector_to_parameters(theta, temp_model.parameters())
        full_inp = [temp_model, inputs, outputs, kl_fn] + [v] + [regu_coef]
        Hvp = Fvp(*full_inp)
        # e = time.time()
        # print ("Hvp time: ", (e-s))
        if return_model:
            return Hvp, temp_model
        return Hvp
    return Fvp_fn

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
