import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set()
sns.set_style("white")
sns.set_context("paper")

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
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

fig = plt.figure(figsize=(3.25, 2.5))


bs = np.load("bs.npy")
print (bs)

sample_colors = ['xkcd:dark orange', 'xkcd:orange', 'xkcd:light orange']
shrunk_colors = ['xkcd:dark blue', 'xkcd:blue', 'xkcd:light blue']
i = 0
for p in [100, 50, 25]: #, 250]:
    diffs_shrunk = np.load("diffs_shrunk_p"+str(p)+".npy") / p
    diffs_sample = np.load("diffs_sample_p"+str(p)+".npy") / p

    print (diffs_shrunk)
    print (diffs_sample)
    plt.semilogy(bs, diffs_sample, label='Sample $p='+str(p)+'$', ls='dashed', linewidth=2.5, color=sample_colors[i])
    plt.semilogy(bs, diffs_shrunk, label='Shrunk $p='+str(p)+'$', ls='solid', linewidth=2.5, color=shrunk_colors[i])
    # plt.semilogy(bs, diffs_sample, label='Sample ' + str(p))
    # plt.semilogy(bs, diffs_shrunk, label='Shrunk ' + str(p))
    i += 1

plt.xlabel("Batch Size")
plt.ylabel("Fisher Error")
plt.legend()
plt.tight_layout()
sns.despine()

plt.savefig("fisher_errors.pdf", bbox_inches='tight')
