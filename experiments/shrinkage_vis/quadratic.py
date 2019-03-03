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

# from fisher.optim.hvp_utils import build_Fvp, mean_kl_multinomial, eval_F, eval_H
from fisher.optim.hvp_utils import eval_F, mean_kl_multinomial
from fisher.utils import lanczos

class Model(nn.Module):

    def __init__(self, p=20):
        super(Model, self).__init__()
        self.fc = nn.Linear(p, 1, bias=False)

    def forward(self, X):
        # print (X.shape)
        y = self.fc(X)
        return torch.sigmoid(y)


# def generate_data(n=1000, d=2):
#     X, y = make_regression(n_samples=n,
#                            n_features=d,
#                            n_informative=d,
#                            n_targets=1,
#                            random_state=0)
#     return X, y

def generate_data(n=1000, d=50, seed=0):
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
                               random_state=seed)
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

    import fisher.optim as fisher_optim
    opt = fisher_optim.NGD(model.parameters(), curv_type='fisher', lr=0.01)


    loss_fn = torch.nn.BCELoss()
    # model.fc.weight.data = torch.FloatTensor([[-7.5, 7.5]])
    X, y = data
    bs, dim = X.shape

    Xfull, yfull = full_data
    Xvarfull = Variable(torch.from_numpy(Xfull)).float()
    yvarfull = Variable(torch.from_numpy(yfull)).float()

    # opt.zero_grad()
    # output = model(Xvarfull)
    # loss = loss_fn(output, yvarfull.reshape_as(output))
    # Hfull = eval_F(model.parameters(), loss) # Xvarfull, yvarfull, mean_kl_multinomial, damping=0.0)
    Hfull = eval_F(model, Xvarfull, yvarfull, mean_kl_multinomial, damping=0.0).numpy()

    trace = [tuple(model.fc.weight.data.numpy().squeeze())]

    for iter in range(20):
        Xvar = Variable(torch.from_numpy(X)).float()
        yvar = Variable(torch.from_numpy(y)).float()

        opt.zero_grad()

        # output = model(Xvar)
        # loss = loss_fn(output, yvar.reshape_as(output))
        # loss.backward()

        # H = eval_H(model.parameters(), loss) #, Xvar, yvar, mean_kl_multinomial, damping=0.0)
        H = eval_F(model, Xvar, yvar, mean_kl_multinomial, damping=0.0).numpy()
        w, v = np.linalg.eig(H)
        # print ("Eigs vs trace: ", w, np.sum(w), np.trace(H))
        rho, D = lanczos.estimate_shrinkage_fix(w, dim, bs)
        # print ("Hess: ", H)
        # print ("D, rho: ", D, rho)
        # print ("True D: ", np.trace(H) / 2)
        Hshrunk = (1-rho)*H + rho * np.eye(dim)*D

        w1_true = np.linalg.eig(Hfull)[0][0]
        w1_sample = w[0]
        w1_shrunk = np.linalg.eig(Hshrunk)[0][0]

        # print ("Hshrunk: ", Hshrunk)
        diff_shrunk = np.linalg.norm(Hshrunk-Hfull, 'fro')
        diff_sample = np.linalg.norm(H-Hfull, 'fro')

        diff_shrunk_eig = np.abs(w1_true - w1_shrunk)
        diff_sample_eig = np.abs(w1_true - w1_sample)

        print ("Diff shrunk/sample fro: ", diff_shrunk, diff_sample, diff_shrunk < diff_sample, diff_sample - diff_shrunk)
        print ("Diff shrunk/sample eig: ", diff_shrunk_eig, diff_sample_eig, diff_shrunk_eig < diff_sample_eig, diff_sample_eig - diff_shrunk_eig)
        return diff_shrunk, diff_sample

    trace_dict[algo] = trace

    # plot_contours(X, y, model, loss_fn, traces=trace_dict)


if __name__ == "__main__":

    seeds = list(range(20))
    bs = [5, 10, 100, 250, 500, 1000]
    ps = [25, 50, 100]
    np.save("bs.npy", np.array(bs))

    diffs_shrunk = np.empty((len(seeds), len(ps), len(bs)))
    diffs_sample = np.empty((len(seeds), len(ps), len(bs)))

    for si in range(len(seeds)):
        s = seeds[si]
        for pi in range(len(ps)):
            p = ps[pi]
            for bi in range(len(bs)):
                b = bs[bi]
                print ("seed=", s, ", p=", p, ", batch=", b)
                X, y = generate_data(n=b, d=p, seed=s)
                Xfull, yfull = generate_data(n=100000, d=p, seed=s)
                diff_shrunk, diff_sample = fit((X, y), (Xfull, yfull), p)

                diffs_shrunk[si,pi,bi] = diff_shrunk
                diffs_sample[si,pi,bi] = diff_sample

    np.save('diffs_shrunk_noise1_fix.npy', diffs_shrunk)
    np.save('diffs_sample_noise1_fix.npy', diffs_sample)
