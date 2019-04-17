import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import numpy as np
import seaborn as sns
import os

import gen_batch_ids

sns.set()
sns.set_style("whitegrid", {'grid.linestyle': '--'})

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

def gen_xs(epoch_ids):
    unique, counts = np.unique(epoch_ids, return_counts=True)
    num_dict = dict(zip(unique, counts))
    xs = []
    for i in range(1, max(epoch_ids)+1):
        count = num_dict[i]
        xs_i = np.linspace(i-1, i, count+1)
        xs.extend(xs_i)
    return np.array(xs)

def load_from_path(fpath, seeds=[0], compile='median'):
    ngd_cg_seeds = []
    for s in seeds:
        seeded_path = fpath % s
        data_seed = np.load(seeded_path)
        ngd_cg_seeds.append(data_seed[np.newaxis,:])
    ngd_cg_cat = np.concatenate(ngd_cg_seeds, axis=0)
    if compile == 'mean':
        ngd_cg_mean = np.mean(ngd_cg_cat, axis=0)
    elif compile == 'median':
        ngd_cg_mean = np.median(ngd_cg_cat, axis=0)

    maxs = []
    for i in range(ngd_cg_cat.shape[0]):
        maxs.append(max(ngd_cg_cat[i]))
    maxs = np.array(maxs)

    max_mean = np.mean(maxs)
    max_std = np.std(maxs)

    stdev = np.std(ngd_cg_cat, axis=0)
    return (ngd_cg_mean, stdev), (max_mean, max_std)

def plot(tag='mlp', epochs=10, bs=250, subtag='batch', lr='0.001', file='data', seeds=[0, 1, 2]):
    try:
        os.makedirs("results/"+str(tag)+"/plots/" + subtag)
    except:
        pass

    epoch_ids_path = "results/meta/epoch"+str(epochs)+"_ids_batch"+str(bs)+".npy"
    if not os.path.isfile(epoch_ids_path):
        print ("Generating batch ids")
        gen_batch_ids.generate_batch_ids(epochs, batch_size)

    epoch_ids = np.load(epoch_ids_path)
    xs = gen_xs(epoch_ids)
    print ("Batch size: ", bs)

    fname = "results/"+str(tag)+"/adam/optim_adaptive/shrunk_false/batch_size_"+str(bs)+"/lr_"+str(lr)+"/%s/data.npy"
    (adam, adam_std), (adam_max_mean, adam_max_std) = load_from_path(fname, seeds=seeds)
    print ("adam: ", adam_max_mean, "+/-", adam_max_std)

    fname = "results/"+str(tag)+"/ngd/optim_adaptive/shrunk_false/batch_size_"+str(bs)+"/lr_"+str(lr)+"/%s/data.npy"
    (ngd, ngd_std), (ngd_max_mean, ngd_max_std) = load_from_path(fname, seeds=seeds)
    print ("ngd: ", ngd_max_mean, "+/-", ngd_max_std)

    fname = "results/"+str(tag)+"/natural_adam/optim_adaptive/shrunk_false/batch_size_"+str(bs)+"/lr_"+str(lr)+"/betas0.1_0.1/%s/data.npy"
    (natural_adam_opt, natural_adam_opt_std), (natural_adam_max_mean, natural_adam_max_std) = load_from_path(fname, seeds=seeds)
    print ("natural_adam_opt: ", natural_adam_max_mean, "+/-", natural_adam_max_std)

    fig = plt.figure(figsize=(4, 3))

    plt.plot(xs, natural_adam_opt, label="FANG-Adam$^*$", ls='solid', color='xkcd:fuchsia', zorder=1002, alpha=0.9)
    # ax.fill_between(xs, natural_adam_opt-natural_adam_opt_std, natural_adam_opt+natural_adam_opt_std, alpha=0.2, color='xkcd:fuchsia', zorder=1000)

    plt.plot(xs, ngd, label="NGD", ls='solid', color='#2F4F4F', alpha=0.9)
    # ax.fill_between(xs, ngd-ngd_std, ngd+ngd_std, alpha=0.2, color='#2F4F4F')

    plt.plot(xs, adam, label="Adam", ls='solid', color='#696969', alpha=0.9)
    # ax.fill_between(xs, adam-adam_std, adam+adam_std, alpha=0.2, color='#696969')

    ylims=(90.0, 98.5)
    plt.ylim(ylims)

    plt.ylabel("Accuracy")
    plt.xlabel("Batch " + str(bs))

    sns.despine()

    plt.legend()

    plt.tight_layout()
    plt.savefig("results/"+str(tag)+"/plots/"+ subtag +"/full.pdf", bbox_inches='tight')


epochs = 5
batch_size = 500
tag='mlp_mnist/sample_experiment/'
subtag='adam_ngd_fang_comparison'
plot(tag=tag, epochs=epochs, bs=batch_size, subtag=subtag, seeds=[0])
