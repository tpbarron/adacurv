import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.multiprocessing as mp
from torch.multiprocessing import Pool


def three_sin(x):
    return torch.sum(torch.sin(torch.sin(torch.sin(x))))

def Rop(ys, xs, vs):
    ws = ys.clone().detach().requires_grad_(True)
    gs = torch.autograd.grad(ys, xs, grad_outputs=ws, create_graph=True, retain_graph=True, allow_unused=False)[0]
    re = torch.autograd.grad(gs, ws, grad_outputs=vs, create_graph=True, retain_graph=True, allow_unused=False)[0]
    return re

def Hvp_RopLop(f, x, v):
    df_dx = torch.autograd.grad(f, x, create_graph=True, retain_graph=True)[0]
    Hv = Rop(df_dx, x, v)
    return Hv

def GNvp_RopLop(f, z, x, v):
    print ("Gauss-Newton")
    # ys, zs, xs
    vec = Variable(v, requires_grad=False)

    grads_z = torch.autograd.grad(f, z, create_graph=True)[0]
    hjv = Rop(grads_z, x, vec)
    jhjv = torch.autograd.grad(z, x, hjv)[0]
    print (hjv, jhjv.shape)
    return jhjv, hjv

def Hvp_dbl_bp(f, x, v):
    # v = Variable(v, requires_grad=False)
    grad_fo = torch.autograd.grad(f, x, create_graph=True)[0]
    h = torch.sum(grad_fo * v)
    hvp = torch.autograd.grad(h, x, create_graph=True, retain_graph=True)[0]
    return hvp.data

def compute_hvp(args): #inp_val, vec_val):
    inp_val, vec_val = args
    x = Variable(torch.FloatTensor([inp_val]), requires_grad=True)
    v = Variable(torch.FloatTensor([vec_val]), requires_grad=False)
    f = three_sin(x)
    return Hvp_dbl_bp(f, x, v)

def test(n=10000):
    n_cpus = mp.cpu_count()
    print ("num cpus: ", n_cpus)
    p = Pool(n_cpus)

    import time
    # s1 = time.time()
    data = []
    for i in range(n):
        inp_val = np.random.random(size=10)
        vec_val = np.random.random(size=10)
        data.append((inp_val, vec_val))
    #
    # res = p.map(compute_hvp, data)
    # e1 = time.time()
    # print ("Time 1: ", (e1-s1))

    s2 = time.time()
    for i in range(n):
        inp_val, vec_val = data[i]
        inp = Variable(torch.FloatTensor([inp_val]), requires_grad=True)
        v = Variable(torch.FloatTensor([vec_val]), requires_grad=False)
        z = three_sin(inp)
        l = F.mse_loss(z, torch.zeros_like(z))
        # hvp_rop_lop = Hvp_RopLop(f, inp, v)
        # print ("hvp: ", hvp_rop_lop.data)
        # hvp_dbl_bp = Hvp_dbl_bp(l, inp, v)
        # print ("hvp: ", hvp_dbl_bp.data)
        # print ("hvp: ", hvp_rop_lop.data, hvp_dbl_bp.data)
        gnvp_roplop = GNvp_RopLop(l, z, inp, v)
    e2 = time.time()
    print ("Time 2: ", (e2-s2))


if __name__ == "__main__":
    test(n = 1)
