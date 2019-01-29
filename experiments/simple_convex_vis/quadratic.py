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

from fisher.optim.hvp_utils import build_Fvp, mean_kl_multinomial

import seaborn as sns

sns.set()
sns.set_style("white")

s = 13
np.random.seed(s)
torch.manual_seed(s)

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
                               flip_y=0.1,
                               class_sep=1.0,
                               hypercube=True,
                               shift=0.0,
                               scale=1.0,
                               shuffle=True,
                               random_state=11)
    return X, y

def plot_contours(Xdata, ydata, model, loss_fn, traces=None):
    Xvar = Variable(torch.from_numpy(Xdata)).float()
    yvar = Variable(torch.from_numpy(ydata)).float()

    # x = np.arange(-15.0, 15.1, 1.0)
    # y = np.arange(-15.0, 15.1, 1.0)

    x = np.arange(-5.0, 1.1, 0.1)
    y = np.arange(-0.0, 3.6, 0.1)

    # x = np.arange(-2.1, 5.1, 0.1)
    # y = np.arange(-0.0, 2.5, 0.1)
    X, Y = np.meshgrid(x, y)
    Z = np.empty_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            model.fc.weight.data = torch.FloatTensor([[X[i,j], Y[i,j]]])
            output = model(Xvar)
            loss = loss_fn(output, yvar)
            Z[i,j] = float(loss.data)
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
    CS = ax.contour(X, Y, Z, levels=[0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 15.0, 20.0])
    ax.clabel(CS, inline=1, fontsize=10, fmt='%1.2f')

    cind = 0
    zdict = dict(natural_adam=1001, natural_amsgrad=1000, natural_adagrad=999, ngd=998) #'black', 'red', 'blue', 'green', 'gray', 'yellow']
    colors = dict(natural_adam='xkcd:fuchsia', natural_amsgrad="xkcd:orange", natural_adagrad="xkcd:blue", ngd="#2F4F4F") #'black', 'red', 'blue', 'green', 'gray', 'yellow']
    label_map=dict(sgd="SGD", adam="Adam", ngd="NGD", natural_adam="FANG-Adam$^*$", natural_amsgrad="FANG-AMSGrad$^*$", natural_adagrad="FANG-Adagrad$^*$")
    if traces is not None:
        for k, v in traces.items():
            lines = []
            tr = v
            for i in range(len(tr)-1):
                lines.append([tr[i], tr[i+1]])

            # ax.scatter(*zip(*tr), marker='o', c=colors[cind], s=4, label=label_map[k])

            # This is not visible but used to get legend
            lc = mc.LineCollection(lines, color=colors[k], linewidths=1.5, label=label_map[k], visible=False)
            ax.add_collection(lc)

            for line in lines:
                arrow = plt.arrow(line[0][0],
                          line[0][1],
                          line[1][0]-line[0][0],
                          line[1][1]-line[0][1],
                          color=colors[k],
                          length_includes_head=True,
                          head_width=0.15,
                          head_length=0.15,
                          linewidth=1.0,
                          zorder=zdict[k],
                          alpha=0.8)

            cind += 1

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

    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig('contour_small2.pdf')
    # plt.savefig('contour_large.pdf')

def fit(data):
    model = Model()
    algos = ['natural_adam', 'natural_amsgrad', 'natural_adagrad', 'ngd']
    # algos = ['ngd']
    # algos = ['natural_adagrad', 'natural_adam', 'natural_amsgrad']
    trace_dict = {}
    for algo in algos:

        if algo in ['ngd', 'natural_amsgrad', 'natural_adagrad', 'natural_adam']:
            import fisher.optim as fisher_optim
            fisher_lr = 0.002
            if algo == 'ngd':
                opt = fisher_optim.NGD(model.parameters(),
                                       lr=fisher_lr,
                                       shrunk=False,
                                       lanczos_iters=1,
                                       batch_size=1000)
            elif algo == 'natural_adam':
                opt = fisher_optim.NaturalAdam(model.parameters(),
                                               lr=fisher_lr,
                                               shrunk=False,
                                               lanczos_iters=0,
                                               batch_size=1000,
                                               betas=(0.1, 0.1),
                                               assume_locally_linear=False)
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

        loss_fn = torch.nn.BCELoss()
        model.fc.weight.data = torch.FloatTensor([[-12.5, 2.0]])
        X, y = data

        trace = [tuple(model.fc.weight.data.numpy().squeeze())]

        for iter in range(20):
            Xvar = Variable(torch.from_numpy(X)).float()
            yvar = Variable(torch.from_numpy(y)).float()

            opt.zero_grad()

            output = model(Xvar)
            loss = loss_fn(output, yvar)
            loss.backward()

            if algo in ['ngd', 'natural_amsgrad', 'natural_adagrad', 'natural_adam']:
                Fvp_fn = build_Fvp(model, Xvar, yvar, mean_kl_multinomial)
                opt.step(Fvp_fn)
            else:
                opt.step()

            trace.append(tuple(model.fc.weight.data.numpy().squeeze()))
            print (loss)

        trace_dict[algo] = trace

    plot_contours(X, y, model, loss_fn, traces=trace_dict)


if __name__ == "__main__":
    X, y = generate_data(n=500)
    print (X.shape, y.shape)
    fit((X, y))
