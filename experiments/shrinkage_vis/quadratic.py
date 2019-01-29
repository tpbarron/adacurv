from itertools import count
import sys

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

from fisher.optim.hvp_utils import build_Fvp, mean_kl_multinomial, eval_F
from fisher.utils import lanczos

class Model(nn.Module):

    def __init__(self, p=20):
        super(Model, self).__init__()
        self.fc = nn.Linear(p, 1, bias=False)

    def forward(self, X):
        print (X.shape)
        y = self.fc(X)
        return torch.sigmoid(y)


# def generate_data(n=1000, d=2):
#     X, y = make_regression(n_samples=n,
#                            n_features=d,
#                            n_informative=d,
#                            n_targets=1,
#                            random_state=0)
#     return X, y

def generate_data(n=1000, d=50):
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
                               scale=5.0,
                               shuffle=True,
                               random_state=int(sys.argv[1]))
    return X, y

def plot_contours(Xdata, ydata, model, loss_fn, traces=None):
    Xvar = Variable(torch.from_numpy(Xdata)).float()
    yvar = Variable(torch.from_numpy(ydata)).float()

    x = np.arange(-8.0, 6.1, 0.1)
    y = np.arange(-2.0, 8.1, 0.1)
    # x = np.arange(-2.5, 5.6, 0.1)
    # y = np.arange(-1.0, 3.6, 0.1)
    X, Y = np.meshgrid(x, y)
    Z = np.empty_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            model.fc.weight.data = torch.FloatTensor([[X[i,j], Y[i,j]]])
            output = model(Xvar)
            loss = loss_fn(output, yvar)
            Z[i,j] = float(loss.data)
    # fig = plt.figure(figsize=(4, 3))
    fig, ax = plt.subplots(figsize=(4, 3))
    CS = ax.contour(X, Y, Z, levels=[0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0])
    ax.clabel(CS, inline=1, fontsize=10, fmt='%1.2f')

    cind = 0
    colors = ['black', 'red', 'blue', 'green', 'gray']
    label_map=dict(sgd="SGD", ngd="NGD", natural_adam="FANG-Adam", natural_amsgrad="FANG-Ams", natural_adagrad="FANG-Ada")
    if traces is not None:
        for k, v in traces.items():
            lines = []
            tr = v
            for i in range(len(tr)-1):
                lines.append([tr[i], tr[i+1]])

            # ax.scatter(*zip(*tr), marker='o', c=colors[cind], s=4, label=label_map[k])

            # This is not visible but used to get legend
            lc = mc.LineCollection(lines, color=colors[cind], linewidths=1.0, label=label_map[k], visible=False)
            ax.add_collection(lc)

            for line in lines:
                arrow = plt.arrow(line[0][0],
                          line[0][1],
                          line[1][0]-line[0][0],
                          line[1][1]-line[0][1],
                          color=colors[cind],
                          length_includes_head=True,
                          head_width=0.1,
                          head_length=0.1,
                          zorder=1000)

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

    plt.legend()
    plt.tight_layout()

    plt.savefig('contour_large.pdf')

def fit(data, full_data, p=20):
    model = Model(p)
    algos = ['ngd'] #'sgd', 'natural_adagrad', 'natural_adam', 'natural_amsgrad', 'ngd']

    trace_dict = {}
    for algo in algos:

        if algo in ['ngd', 'natural_amsgrad', 'natural_adagrad', 'natural_adam']:
            import fisher.optim as fisher_optim
            if algo == 'ngd':
                opt = fisher_optim.NGD(model.parameters(),
                                       lr=0.01,
                                       shrunk=False,
                                       lanczos_iters=1,
                                       batch_size=1000)
            elif algo == 'natural_adam':
                opt = fisher_optim.NaturalAdam(model.parameters(),
                                               lr=0.01,
                                               shrunk=False,
                                               lanczos_iters=0,
                                               batch_size=1000,
                                               betas=(0.1, 0.1),
                                               assume_locally_linear=False)
            elif algo == 'natural_amsgrad':
                opt = fisher_optim.NaturalAmsgrad(model.parameters(),
                                               lr=0.01,
                                               shrunk=False,
                                               lanczos_iters=0,
                                               batch_size=1000,
                                               betas=(0.1, 0.1),
                                               assume_locally_linear=False)
            elif algo == 'natural_adagrad':
                opt = fisher_optim.NaturalAdagrad(model.parameters(),
                                               lr=0.01,
                                               shrunk=False,
                                               lanczos_iters=0,
                                               batch_size=1000,
                                               assume_locally_linear=False)
        else:
            opt = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

        loss_fn = torch.nn.BCELoss()
        # model.fc.weight.data = torch.FloatTensor([[-7.5, 7.5]])
        X, y = data
        bs, dim = X.shape

        Xfull, yfull = full_data
        Xvarfull = Variable(torch.from_numpy(Xfull)).float()
        yvarfull = Variable(torch.from_numpy(yfull)).float()
        Hfull = eval_F(model, Xvarfull, yvarfull, mean_kl_multinomial, damping=0.0)

        trace = [tuple(model.fc.weight.data.numpy().squeeze())]

        for iter in range(20):
            Xvar = Variable(torch.from_numpy(X)).float()
            yvar = Variable(torch.from_numpy(y)).float()

            opt.zero_grad()

            output = model(Xvar)
            loss = loss_fn(output, yvar.reshape_as(output))
            loss.backward()

            if algo in ['ngd', 'natural_amsgrad', 'natural_adagrad', 'natural_adam']:
                H = eval_F(model, Xvar, yvar, mean_kl_multinomial, damping=0.0)
                w, v = np.linalg.eig(H)
                # print ("Eigs vs trace: ", w, np.sum(w), np.trace(H))
                rho, D = lanczos.estimate_shrinkage(w, dim, bs)
                # print ("Hess: ", H)
                print ("D, rho: ", D, rho)
                # print ("True D: ", np.trace(H) / 2)
                Hshrunk = (1-rho)*H + rho * np.eye(dim)*D
                # print ("Hshrunk: ", Hshrunk)
                diff_shrunk = np.linalg.norm(Hshrunk-Hfull, 'fro')
                diff_sample = np.linalg.norm(H-Hfull, 'fro')
                print ("Diff shrunk/sample: ", diff_shrunk, diff_sample, diff_shrunk < diff_sample, diff_sample - diff_shrunk)
                return diff_shrunk, diff_sample
                Fvp_fn = build_Fvp(model, Xvar, yvar, mean_kl_multinomial)
                opt.step(Fvp_fn)
            else:
                opt.step()

            trace.append(tuple(model.fc.weight.data.numpy().squeeze()))
            print (loss)

        trace_dict[algo] = trace

    plot_contours(X, y, model, loss_fn, traces=trace_dict)


if __name__ == "__main__":

    bs = [5, 10, 100, 250, 500, 1000]
    np.save("bs.npy", np.array(bs))
    for p in [25, 50, 100]: #[10, 20, 25, 50, 100]:
        diffs_shrunk = []
        diffs_sample = []
        for i in bs:
            print ("bs: ", i)
            X, y = generate_data(n=i, d=p)
            Xfull, yfull = generate_data(n=1000000, d=p)
            diff_shrunk, diff_sample = fit((X, y), (Xfull, yfull), p)
            diffs_shrunk.append(diff_shrunk)
            diffs_sample.append(diff_sample)

        np.save('diffs_shrunk_p'+str(p)+'.npy', np.array(diffs_shrunk))
        np.save('diffs_sample_p'+str(p)+'.npy', np.array(diffs_sample))
