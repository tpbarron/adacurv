import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import numpy as np
import seaborn as sns
import os
import csv

from scipy.ndimage.filters import gaussian_filter1d

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

def load_from_path(fpath, seeds=[0], compile='mean', smooth=True):
    ngd_cg_seeds = []

    min_complete = np.inf
    for s in seeds:
        seeded_path = fpath % s
        print (seeded_path)
        with open(seeded_path, 'r') as f:
            reader = csv.reader(f)
            row1 = next(reader)  # gets the first line
            cind = row1.index('stoc_pol_mean')

            ngd_cg = np.loadtxt(seeded_path, delimiter=',', skiprows=2, usecols=(cind,))
            if len(ngd_cg) < min_complete:
                min_complete = len(ngd_cg)
            ngd_cg_seeds.append(ngd_cg[np.newaxis,:])
            print (ngd_cg.shape)
    for i in range(len(ngd_cg_seeds)):
        ngd_cg_seeds[i] = ngd_cg_seeds[i][:,:min_complete]
    ngd_cg_cat = np.concatenate(ngd_cg_seeds, axis=0)
    if compile == 'mean':
        ngd_cg_mean = np.mean(ngd_cg_cat, axis=0)
    elif compile == 'median':
        ngd_cg_mean = np.median(ngd_cg_cat, axis=0)
    elif compile == 'max':
        ngd_cg_mean = np.max(ngd_cg_cat, axis=0)
    stdev = np.std(ngd_cg_cat, axis=0)

    if smooth:
        ngd_cg_mean = gaussian_filter1d(ngd_cg_mean, sigma=1)
        stdev = gaussian_filter1d(stdev, sigma=1)

    return ngd_cg_mean, stdev

def plot(tag='mlp',
         subtag='batch',
         env='HalfCheetahBulletEnv-v0',
         lr='0.0',
         file='log',
         bs=5000,
         total_samples=1000000,
         policy='nn_policy',
         lanczos_itr=5,
         seeds=[0, 1, 2, 3, 4],
         show_shrunk=True,
         compile='mean'):
    try:
        os.makedirs("results/"+str(tag)+"/plots/" + subtag)
    except:
        pass

    fpath = "results/"+str(tag)+"/"+str(env)+"/trpo/natural_adam/optim_adaptive/shrunk_true/lanczos_iters_"+str(lanczos_itr)+"/"+policy+"/total_samples_"+str(total_samples)+"/batch_size_"+str(bs)+"/lr_"+str(lr)+"/betas0.1_0.1/%s/logs/"+str(file)+".csv"
    trpo_natural_adam_shrunk_mean, trpo_natural_adam_shrunk_stdev = load_from_path(fpath, seeds=seeds, compile=compile)

    fpath = "results/"+str(tag)+"/"+str(env)+"/trpo/ngd/optim_adaptive/shrunk_false/"+policy+"/total_samples_"+str(total_samples)+"/batch_size_"+str(bs)+"/lr_"+str(lr)+"/%s/logs/"+str(file)+".csv"
    trpo_mean, trpo_stdev = load_from_path(fpath, seeds=seeds, compile=compile)

    lw = 1.0
    plt.plot(trpo_natural_adam_shrunk_mean, label="FANG-Adam$^{*(s)}$-TRPO", ls='solid', linewidth=lw, color='xkcd:fuchsia')
    plt.gca().fill_between(np.arange(len(trpo_natural_adam_shrunk_mean)), trpo_natural_adam_shrunk_mean-trpo_natural_adam_shrunk_stdev, trpo_natural_adam_shrunk_mean+trpo_natural_adam_shrunk_stdev, alpha=0.1, color='xkcd:fuchsia', zorder=1000)

    plt.plot(trpo_mean, label="TRPO", ls='solid', linewidth=lw, color='#2F4F4F')
    plt.gca().fill_between(np.arange(len(trpo_mean)), trpo_mean-trpo_stdev, trpo_mean+trpo_stdev, alpha=0.1, color='#2F4F4F')

    plt.ylabel("Reward")
    plt.xlabel(env[:-12])


    plt.tight_layout()
    plt.legend()

    sns.despine()

    plt.savefig("results/"+str(tag)+"/plots/"+ subtag +"/"+str(env)+"_bs"+str(bs)+"_rewards_"+compile+".pdf")

# tag = 'pybullet_sample_mode_trpo'
tag = 'quick_start/'
subtag = 'hopper_exp'
plot(tag=tag, env='HopperBulletEnv-v0', subtag=subtag, compile='mean', lr='0.005')
