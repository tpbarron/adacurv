from itertools import count

from sklearn.datasets import make_regression, make_classification
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import collections  as mc

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd
from torch.autograd import Variable
import torch.nn.functional as F

from fisher.optim.hvp_utils import build_Fvp, mean_kl_multinomial, eval_F, build_eval_F
from fisher.utils.convert_gradients import gradients_to_vector, vector_to_gradients

import seaborn as sns

# np.random.seed(7)
# torch.manual_seed(7)

sns.set()
sns.set_style("white")

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.fc = nn.Linear(2, 1, bias=False)

    def forward(self, X):
        y = self.fc(X)
        return torch.sigmoid(y)


# def generate_data(n=1000, d=2):
#     X, y = make_regression(n_samples=n,
#                            n_features=d,
#                            n_informative=d,
#                            n_targets=1,
#                            random_state=0)
#     return X, y

def generate_data(n=500, d=2):
    X, y = make_classification(n_samples=n,
                               n_features=d,
                               n_informative=d,
                               n_redundant=0,
                               n_repeated=0,
                               n_classes=2,
                               n_clusters_per_class=2,
                               flip_y=0.0, # 0.01
                               class_sep=1.0,
                               hypercube=True,
                               shift=0.0,
                               scale=1.0,
                               shuffle=True,
                               random_state=11)
    return X, y


def eigsorted(cov):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    return vals[order], vecs[:,order]

def plot_contours(Xdata, ydata, model, loss_fn):
    Xvar = Variable(torch.from_numpy(Xdata)).float()
    yvar = Variable(torch.from_numpy(ydata)).float()

    x = np.arange(-25.0, 25.1, 1.0)
    y = np.arange(-25.0, 25.1, 1.0)

    point = model.fc.weight.data.clone()

    # x = np.arange(-4.0, 7.1, 0.1)
    # y = np.arange(-1.0, 4.1, 0.1)

    # x = np.arange(-2.5, 4.6, 0.1)
    # y = np.arange(-0.5, 2.5, 0.1)
    X, Y = np.meshgrid(x, y)
    Z = np.empty_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            model.fc.weight.data = torch.FloatTensor([[X[i,j], Y[i,j]]])
            output = model(Xvar)
            loss = loss_fn(output, yvar)
            Z[i,j] = float(loss.data)

    model.fc.weight.data = point

    # fig = plt.figure(figsize=(4, 3))
    plt.rc('font', family='serif')
    plt.rc('text', usetex=True)

    SMALL_SIZE = 8
    MEDIUM_SIZE = 10
    BIGGER_SIZE = 12

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    fig, ax = plt.subplots(figsize=(3.75, 2.75))
    CS = ax.contour(X, Y, Z, levels=[0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0])
    ax.clabel(CS, inline=1, fontsize=10, fmt='%1.2f')


def plot_grads_test(Xdata, ydata, model, loss_fn, point, grads, Ft, algo='ngd'):
    cind = 0 if algo == 'ngd' else 2
    colors = ['blue', 'red', 'orange', 'green', 'gray', 'yellow']
    # label_map=dict(sgd="SGD", adam="Adam", ngd="NGD", natural_adam="FANG-Adam$^*$", natural_amsgrad="FANG-AMSGrad$^*$", natural_adagrad="FANG-Adagrad$^*$")

    from matplotlib.patches import Ellipse
    # w, v = eigsorted(np.linalg.pinv(Ft))
    w, v = eigsorted(Ft)
    print ("eigs: ", w)
    # theta = np.degrees(np.arctan2(*v[:,0][::-1]))
    vx, vy = v[:,0][0], v[:,0][1]
    theta = np.degrees(np.arctan2(vy, vx))
    w /= max(w)
    el = Ellipse(point, *w, theta, color=colors[cind], alpha=0.25)
    plt.gca().add_artist(el)

    lines = []
    grads_alg = grads #[key]
    pt1 = point #np.array(traces[key][i])
    max_norm = 0.0
    # print (grads_alg)
    for gi in range(0, len(grads_alg), 1):
        grad = grads_alg[gi]
        grad_norm = np.linalg.norm(grad)
        if grad_norm > max_norm:
            max_norm = grad_norm

    for gi in range(0, len(grads_alg), 1):
        grad = grads_alg[gi]
        # print (grad)
        # grad_unit = grad / np.linalg.norm(grad)
        pt2 = pt1 + -1. * grad / max_norm #_unit
        # print (pt1, pt2)
        lines.append([list(pt1), list(pt2)])

    # print ("Arrows")
    for line in lines:
        try:
            # print ("Line: ", line)
            arrow = plt.arrow(line[0][0],
                      line[0][1],
                      line[1][0]-line[0][0],
                      line[1][1]-line[0][1],
                      color=colors[cind],
                      length_includes_head=True,
                      head_width=0.05,
                      head_length=0.05,
                      linewidth=0.25,
                      alpha=0.1,
                      zorder=1001)
        except:
            pass
    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False) # labels along the bottom edge are off
    plt.tick_params(
        axis='y',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        left=False,        # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelleft=False)   # labels along the bottom edge are off


def compute_sample_gradients(Xvar, yvar, model, opt, loss_fn, algo):
    n = 10
    grads = []
    if algo in ['ngd', 'natural_amsgrad', 'natural_adagrad', 'natural_adam']:
        H = eval_F(model, Xvar, yvar, mean_kl_multinomial, damping=0.0)
        Hinv = np.linalg.pinv(H)
    for i in range(0, Xvar.shape[0], n):
        Xvar_sub = Xvar[i:i+n]
        yvar_sub = yvar[i:i+n]
        # print (Xvar.shape, Xvar_sub.shape)
        opt.zero_grad()
        output = model(Xvar_sub)
        loss = loss_fn(output, yvar_sub)
        loss.backward()

        g = gradients_to_vector(model.parameters())
        if algo in ['ngd', 'natural_amsgrad', 'natural_adagrad', 'natural_adam']:
            g = np.copy(Hinv @ g.numpy())
        grads.append(list(g))

    grads = np.array(grads)
    return grads

def fit(data):
    model = Model()
    algos = ['ngd', 'natural_adam'] #, 'natural_adam'] #, 'natural_amsgrad', 'natural_adagrad', 'natural_adam'] # 'natural_amsgrad', 'ngd']
    # algos = ['natural_adagrad', 'natural_adam', 'natural_amsgrad']
    trace_dict = {}
    grad_dict = {}
    avg_grad_dict = {}

    loss_fn = torch.nn.BCELoss()
    # model.fc.weight.data = torch.FloatTensor([[-7., 6.5]])
    X, y = data
    plot_contours(X, y, model, loss_fn)


    for algo in algos:

        if algo in ['ngd', 'natural_amsgrad', 'natural_adagrad', 'natural_adam']:
            import fisher.optim as fisher_optim
            fisher_lr = 0.005
            if algo == 'ngd':
                opt = fisher_optim.NGDExact(model.parameters(),
                                       lr=fisher_lr,
                                       shrunk=False,
                                       lanczos_iters=1,
                                       batch_size=1000)
            elif algo == 'natural_adam':
                opt = fisher_optim.NaturalAdamExact(model.parameters(),
                                               lr=fisher_lr,
                                               shrunk=False,
                                               lanczos_iters=0,
                                               batch_size=1000,
                                               betas=(0.1, 0.1))
                                               # assume_locally_linear=False)
            elif algo == 'natural_amsgrad':
                opt = fisher_optim.NaturalAmsgrad(model.parameters(),
                                               lr=fisher_lr,
                                               shrunk=False,
                                               lanczos_iters=0,
                                               batch_size=1000,
                                               betas=(0.1, 0.1),
                                               assume_locally_linear=False)
            elif algo == 'natural_adagrad':
                opt = fisher_optim.NaturalAdagrad(model.parameters(),
                                               lr=fisher_lr,
                                               shrunk=False,
                                               lanczos_iters=0,
                                               batch_size=1000,
                                               assume_locally_linear=False)
        else:
            if algo == 'sgd':
                opt = optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
            elif algo == 'adam':
                opt = optim.Adam(model.parameters(), lr=0.1)

        model.fc.weight.data = torch.FloatTensor([[-5.5, 5.0]])

        trace = [tuple(model.fc.weight.data.numpy().squeeze())]
        grad_history = []
        avg_grad_history = []

        for iter in range(30):
            # idxs = np.random.choice(X.shape[0], size=500, replace=False)
            # Xvar = Variable(torch.from_numpy(X[idxs])).float()
            # yvar = Variable(torch.from_numpy(y[idxs])).float()
            Xvar = Variable(torch.from_numpy(X)).float()
            yvar = Variable(torch.from_numpy(y)).float()

            Xvar_full = Variable(torch.from_numpy(X)).float()
            yvar_full = Variable(torch.from_numpy(y)).float()

            opt.zero_grad()
            output = model(Xvar)
            loss = loss_fn(output, yvar)
            loss.backward()

            if algo in ['ngd', 'natural_amsgrad', 'natural_adagrad', 'natural_adam']:
                # Fvp_fn = build_Fvp(model, Xvar, yvar, mean_kl_multinomial)
                Fvp_fn = build_eval_F(model, Xvar, yvar, mean_kl_multinomial)
                opt.step(Fvp_fn)
                averaged_grad = None #opt.state['ng_prior'].numpy()
            else:
                opt.step()
                averaged_grad = gradients_to_vector(model.parameters())

            grads = compute_sample_gradients(Xvar_full, yvar_full, model, opt, loss_fn, algo)
            point = model.fc.weight.data.numpy().squeeze().copy()
            F = opt.state['Ft']

            if iter in [0, 1, 2, 3, 4, 5]: #, 10, 15, 20]:
                plot_grads_test(X, y, model, loss_fn, point, grads, F, algo=algo)
                # input("")

            print ("pt: ", point)

            trace.append(tuple(model.fc.weight.data.numpy().squeeze()))
            grad_history.append(grads)
            avg_grad_history.append(averaged_grad)

        trace_dict[algo] = trace
        grad_dict[algo] = grad_history
        avg_grad_dict[algo] = avg_grad_history


    plt.legend()
    plt.tight_layout()
    plt.savefig('contour_stochastic_'+algo+'.pdf')
    # plot_grads_test(X, y, model, loss_fn, traces=trace_dict, grads=grad_dict, avg_grads=avg_grad_dict)


if __name__ == "__main__":
    X, y = generate_data(n=2000)
    print (X.shape, y.shape)
    fit((X, y))
