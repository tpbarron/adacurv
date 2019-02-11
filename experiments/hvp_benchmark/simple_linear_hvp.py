import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.multiprocessing as mp
from torch.multiprocessing import Pool


def three_sin(x):
    return torch.sum(torch.sin(torch.sin(torch.sin(x))))

def Rop(ys, xs, vs):
    ws = ys.clone().detach().requires_grad_(True)
    # print ("ws: ", ws)
    gs = torch.autograd.grad(ys, xs, grad_outputs=ws, create_graph=True, retain_graph=True, allow_unused=False)
    gs = torch.cat([g.contiguous().view(-1) for g in gs])
    # print ("gs: ", gs)
    # input("")
    re = torch.autograd.grad(gs, ws, grad_outputs=vs, create_graph=True, retain_graph=True, allow_unused=False)
    # re = torch.cat([r.contiguous().view(-1) for r in re])
    # print ("re: ", re)
    # input("")
    return re

def Hvp_RopLop(f, x, v):
    df_dx = torch.autograd.grad(f, x, create_graph=True, retain_graph=True)[0]
    Hv = Rop(df_dx, x, v)
    return Hv

def GNvp_RopLop(f, z, x, v):
    print ("Gauss-Newton")
    vec = Variable(v, requires_grad=False)
    grads_z = torch.autograd.grad(f, z, create_graph=True, retain_graph=True)[0]
    # print ("grad_z: ", grads_z)
    hjv = Rop(grads_z, x, vec)
    # print ("z: ", z)
    # print ("hjv: ", hjv)
    jhjv = torch.autograd.grad(z, x, grad_outputs=hjv, create_graph=True, retain_graph=True)
    return jhjv #, hjv

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

def test(model):

    inp_val = np.random.random(size=(1, 20))
    vec_val = np.random.random(size=(1, 117))
    inp = Variable(torch.FloatTensor([inp_val]), requires_grad=True)
    v = Variable(torch.FloatTensor([vec_val]), requires_grad=False)
    z = model(inp)
    l = F.mse_loss(z, torch.zeros_like(z))

    gnvp_roplop = GNvp_RopLop(l, z, list(model.parameters()), v)
    print (gnvp_roplop)

class SimpleModel(nn.Module):

    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(20, 5)
        self.fc2 = nn.Linear(5, 2)

    def forward(self, x):
        return self.fc2(torch.sigmoid(self.fc1(x)))


if __name__ == "__main__":
    model = SimpleModel()
    test(model)
