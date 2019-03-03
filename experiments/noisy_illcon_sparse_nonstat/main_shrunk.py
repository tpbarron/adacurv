import os
import numpy as np

import torch
import torch.optim as optim
from torch.autograd import Variable

from fisher.utils.convert_gradients import gradients_to_vector, vector_to_gradients

import quadratic_generator

class Quadratic:
    def __init__(self, Q, x, v=0):
        self.Q = Q
        self.Qvar = Variable(torch.from_numpy(Q), requires_grad=False).float()
        self.x = x
        # self.xvar = Variable(torch.from_numpy(x), requires_grad=False).float()
        self.v = v
        self.dim = self.Q.shape[0]

    def sample(self, n=1):
        """
        generate data
        """
        data = np.empty((self.dim, n))
        for s in range(n):
            xi = np.random.normal(loc=self.x, scale=self.v**2.0)
            data[:, s] = xi
        return data

    def loss_fn(self, x, p):
        px = torch.mean(p - x, dim=1).view(-1, 1)
        return 0.5 * px.t() @ self.Qvar @ px + self.v**2.0 / 2.0 * torch.trace(self.Qvar)
        # return 0.5 * torch.mean(p - x, dim=1).t() @ self.Qvar @ (p - x) + self.v**2.0 / 2.0 * torch.trace(self.Qvar)


import fisher.optim as fisher_optim
def train(args, quad):

    p = torch.nn.Parameter(torch.zeros(args.dimension, 1))

    losses = np.empty((args.iters, 2))

    best_loss = np.inf
    if args.sgd:
        opt = optim.SGD([p], lr=0.01)
    else:
        opt = fisher_optim.Newton([p], lr=0.01, adaptive=args.adaptive, Q=quad.Q)
    for i in range(args.iters):
        opt.zero_grad()
        data = quad.sample(n=args.batch_size)

        x = Variable(torch.from_numpy(data)).float()
        l = quad.loss_fn(x, p)
        l.backward(retain_graph=not args.sgd)

        # Get flat grad
        g = gradients_to_vector([p])
        a = torch.empty_like(g).fill_(1.0-args.grad_sparsity)
        g_sparse = torch.bernoulli(a) * g
        vector_to_gradients(g_sparse, [p])

        if args.sgd:
            opt.step()
        else:
            opt.step(l)

        l = float(l)
        if l < best_loss:
            best_loss = l
        losses[i,0] = l
        losses[i,1] = best_loss
        # print ("Loss (", i, ")", l, best_loss)

    # print ("Best loss: ", best_loss)
    return losses

def build_log_dir(args):
    dir = args.log_dir
    dir = os.path.join(dir, "batch_size_" + str(args.batch_size))
    dir = os.path.join(dir, "iters_" + str(args.iters))
    dir = os.path.join(dir, "dimension_" + str(args.dimension))
    dir = os.path.join(dir, "condition_" + str(args.condition))
    dir = os.path.join(dir, "noise_" + str(args.noise))
    dir = os.path.join(dir, "grad_sparsity_" + str(args.grad_sparsity))

    if args.rotate:
        dir = os.path.join(dir, "rotate_true")
    else:
        dir = os.path.join(dir, "rotate_false")

    if args.adaptive:
        dir = os.path.join(dir, "adaptive_true")
    else:
        dir = os.path.join(dir, "adaptive_false")

    dir = os.path.join(dir, str(args.seed))
    return dir


def compute_true_hessian(args, quad):
    np.random.seed(1)
    torch.manual_seed(1)
    p = torch.nn.Parameter(torch.zeros(args.dimension, 1))
    opt = fisher_optim.Newton([p], lr=0.01, adaptive=args.adaptive, Q=quad.Q)

    opt.zero_grad()
    data = quad.sample(n=1000000) #args.batch_size)

    x = Variable(torch.from_numpy(data)).float()
    l = quad.loss_fn(x, p)
    l.backward(retain_graph=not args.sgd)

    # Get flat grad
    g = gradients_to_vector([p])
    a = torch.empty_like(g).fill_(1.0-args.grad_sparsity)
    g_sparse = torch.bernoulli(a) * g
    vector_to_gradients(g_sparse, [p])
    info = opt.step(l)
    H = info['H']
    return H.data.numpy()

def compute_sample_hessian(args, quad):
    np.random.seed(1)
    torch.manual_seed(1)

    p = torch.nn.Parameter(torch.zeros(args.dimension, 1))
    opt = fisher_optim.Newton([p], lr=0.01, adaptive=args.adaptive, Q=quad.Q)

    opt.zero_grad()
    data = quad.sample(n=args.batch_size)
    print ("data: ", data.shape)

    x = Variable(torch.from_numpy(data)).float()
    l = quad.loss_fn(x, p)
    l.backward(retain_graph=not args.sgd)

    # Get flat grad
    g = gradients_to_vector([p])
    a = torch.empty_like(g).fill_(1.0-args.grad_sparsity)
    g_sparse = torch.bernoulli(a) * g
    vector_to_gradients(g_sparse, [p])
    info = opt.step(l)
    H = info['H']

    return H.data.numpy()

def compute_shrunk_hessian(args, quad):
    np.random.seed(1)
    torch.manual_seed(1)

    p = torch.nn.Parameter(torch.zeros(args.dimension, 1))
    opt = fisher_optim.Newton([p], lr=0.01, adaptive=args.adaptive, Q=quad.Q)

    opt.zero_grad()
    data = quad.sample(n=args.batch_size)

    x = Variable(torch.from_numpy(data)).float()
    l = quad.loss_fn(x, p)
    l.backward(retain_graph=not args.sgd)

    # Get flat grad
    # g = gradients_to_vector([p])
    # a = torch.empty_like(g).fill_(1.0-args.grad_sparsity)
    # g_sparse = torch.bernoulli(a) * g
    # vector_to_gradients(g_sparse, [p])

    info = opt.step(l)
    H = info['H']
    eigs = np.linalg.eigvalsh(H)
    from fisher.utils.lanczos import lanczos_iteration, estimate_shrinkage

    # print (eigs, len(p))
    rho, D = estimate_shrinkage(eigs, len(p), args.batch_size)
    # print ("D: ", D, ", rho:", rho)
    H = H.data.numpy()
    Hshrunk = (1-rho) * H + rho * np.eye(len(p)) * D
    return Hshrunk


def launch_job(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    Q, w = quadratic_generator.generate_quadratic(args.condition, args.dimension, args.rotate)
    x = np.random.randn(args.dimension)
    # np.save("eigs_cond_"+str(args.condition)+".npy", w)

    quad1 = Quadratic(np.copy(Q), x, v=args.noise)
    quad2 = Quadratic(np.copy(Q), x, v=args.noise)
    quad_noiseless = Quadratic(np.copy(Q), x, v=0.0)

    Htrue = compute_true_hessian(args, quad_noiseless)
    print(Htrue.shape)
    Hsample = compute_sample_hessian(args, quad1)
    print (Hsample.shape)
    Hshrunk = compute_shrunk_hessian(args, quad2)
    print(Hshrunk.shape)

    print ("Sample error: ", np.linalg.norm(Htrue - Hsample))
    print ("Shrunk error: ", np.linalg.norm(Htrue - Hshrunk))

    input("")

    path = build_log_dir(args)

    try:
        os.makedirs(path)
    except:
        pass

    print ("Saving to: ", path)
    np.save(os.path.join(path, 'shrunk.npy'), data)

if __name__ == "__main__":

    import arguments
    args = arguments.get_args()

    launch_job(args)
