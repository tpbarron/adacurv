import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import numpy as np
import seaborn as sns
import os

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

def maybe_fix_data(fpath):
    # load data,
    data = np.load(fpath)
    if data.shape[0] in [491]:
        # , 239, 467, 491, 923]:
        print ("Fixing data")
        # resave as backup.
        np.save(fpath+".backup.npy", data)
        # subsample
        # new_data = []
        j = 0
        idx = np.round(np.linspace(0, data.shape[0] - 1, 111)).astype(int)

        #subsamples = np.concatenate([np.array([0]), np.arange(5, 46, 5), np.array([48])])
        # for i in range(10):
        #     new_data.extend(data[subsamples+j])
        #     j += subsamples[-1]

        new_data = data[idx]
        # new_data.append(data[-1])
        np.save(fpath, new_data) #np.array(new_data))

def load_from_path(fpath, seeds=[0], compile='median'):
    ngd_cg_seeds = []
    for s in seeds:
        seeded_path = fpath % s
        # print (seeded_path)
        maybe_fix_data(seeded_path)
        data_seed = np.load(seeded_path)
        # print (data_seed.shape)
        ngd_cg_seeds.append(data_seed[np.newaxis,:])
    ngd_cg_cat = np.concatenate(ngd_cg_seeds, axis=0)
    # print (ngd_cg_cat.shape)
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

def fix_data_seed0(fpath, epoch_ids, epoch_ids_slim):
    # load data.npy.backup.npy
    orig_data = np.load(fpath+".backup.npy")

    unique, counts = np.unique(epoch_ids, return_counts=True)
    num_dict = dict(zip(unique, counts))
    print (unique, counts)
    unique_slim, counts_slim = np.unique(epoch_ids_slim, return_counts=True)
    num_dict_slim = dict(zip(unique_slim, counts_slim))
    print (unique_slim, counts_slim)

    idxs = []
    max_ind = 0
    for i in range(1, max(epoch_ids)+1):
        count_slim = num_dict_slim[i] # num to generate
        count = num_dict[i] # num existing
        idx = np.round(np.linspace(max_ind, max_ind+count-1, count_slim+1)).astype(int)
        # print (idx)

        idxs.extend(list(idx))
        max_ind += count
        # xs_i = np.linspace(i-1, i, count+1)
        # xs.extend(xs_i)

    np_idxs = np.array(idxs)
    print (np_idxs)
    print (orig_data[np_idxs])
    # input("")
    np.save(fpath, orig_data[np_idxs])

def subsample(tag='mlp', bs=250, subtag='batch', lr='0.001', file='data', seeds=[0, 1, 2]):
    try:
        os.makedirs("results/"+str(tag)+"/plots/" + subtag)
    except:
        pass

    print ("bs, seeds:", bs, seeds)

    epoch_ids = np.load("results/meta/epoch_ids_batch"+str(bs)+".npy")
    epoch_ids_slim = np.load("results/meta/epoch_ids_batch_slim"+str(bs)+".npy")

    print (epoch_ids.shape)
    print (epoch_ids_slim.shape)

    xs = gen_xs(epoch_ids_slim)
    print (xs.shape)

    fname = "results/"+str(tag)+"/adam/optim_adaptive/shrunk_false/batch_size_"+str(bs)+"/lr_"+str(lr)+"/0/data.npy"
    fix_data_seed0(fname, epoch_ids, epoch_ids_slim)

    fname = "results/"+str(tag)+"/adagrad/optim_adaptive/shrunk_false/batch_size_"+str(bs)+"/lr_"+str(lr)+"/0/data.npy"
    fix_data_seed0(fname, epoch_ids, epoch_ids_slim)

    fname = "results/"+str(tag)+"/amsgrad/optim_adaptive/shrunk_false/batch_size_"+str(bs)+"/lr_"+str(lr)+"/0/data.npy"
    fix_data_seed0(fname, epoch_ids, epoch_ids_slim)

    fname = "results/"+str(tag)+"/sgd/optim_adaptive/shrunk_false/batch_size_"+str(bs)+"/lr_"+str(lr)+"/0/data.npy"
    fix_data_seed0(fname, epoch_ids, epoch_ids_slim)

    fname = "results/"+str(tag)+"/ngd/optim_adaptive/shrunk_false/batch_size_"+str(bs)+"/lr_"+str(lr)+"/0/data.npy"
    fix_data_seed0(fname, epoch_ids, epoch_ids_slim)

    fname = "results/"+str(tag)+"/natural_adagrad/optim_adaptive/shrunk_false/batch_size_"+str(bs)+"/lr_"+str(lr)+"/0/data.npy"
    fix_data_seed0(fname, epoch_ids, epoch_ids_slim)

    # Reduce LR for nat adam and nat amsgrad
    if bs in [125]:
        lr = 0.0001
    elif bs in [250]:
        lr = 0.0005

    fname = "results/"+str(tag)+"/natural_amsgrad/optim_adaptive/shrunk_false/batch_size_"+str(bs)+"/lr_"+str(lr)+"/betas0.1_0.1/0/data.npy"
    fix_data_seed0(fname, epoch_ids, epoch_ids_slim)

    fname = "results/"+str(tag)+"/natural_adam/optim_adaptive/shrunk_false/batch_size_"+str(bs)+"/lr_"+str(lr)+"/betas0.1_0.1/0/data.npy"
    fix_data_seed0(fname, epoch_ids, epoch_ids_slim)


batch_sizes = [125, 250, 500, 1000]
tag='mlp_mnist_seeds_decaysqrt'

for b in batch_sizes:
    subsample(tag=tag, bs=b, subtag='test', seeds=[0])
