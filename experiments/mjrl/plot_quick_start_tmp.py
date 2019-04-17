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

# fig = plt.figure(figsize=(4, 3))

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
    stdev = np.std(ngd_cg_cat, axis=0)

    if smooth:
        ngd_cg_mean = gaussian_filter1d(ngd_cg_mean, sigma=1)
        stdev = gaussian_filter1d(stdev, sigma=1)

    return ngd_cg_mean, stdev

def plot(tag='mlp',
         subtag='batch',
         env='HalfCheetahBulletEnv-v0',
         lr='0.005',
         file='log',
         bs=5000,
         total_samples=1000000,
         policy='nn_policy',
         lanczos_itr=5,
         seeds=[0, 1, 2, 3, 4],
         show_shrunk=True,
         compile='mean',
         fig_ax=None):
    try:
        os.makedirs("results/"+str(tag)+"/plots/" + subtag)
    except:
        pass

    if fig_ax is not None:
        fig, ax = fig_ax

    # fpath = "results/"+str(tag)+"/"+str(env)+"/ngd/optim_adaptive/shrunk_false/"+policy+"/total_samples_"+str(total_samples)+"/batch_size_"+str(bs)+"/lr_"+str(lr)+"/%s/logs/"+str(file)+".csv"
    # ngd_cg_mean = load_from_path(fpath, seeds=[2])
    #
    # fpath = "results/"+str(tag)+"/"+str(env)+"/trpo/optim_adaptive/shrunk_false/"+policy+"/total_samples_"+str(total_samples)+"/batch_size_"+str(bs)+"/lr_0.0/%s/logs/"+str(file)+".csv"
    # trpo_mean = load_from_path(fpath, seeds=seeds)
    #
    # fpath = "results/"+str(tag)+"/"+str(env)+"/natural_adam/optim_adaptive/shrunk_false/"+policy+"/total_samples_"+str(total_samples)+"/batch_size_"+str(bs)+"/lr_"+str(lr)+"/betas0.1_0.1/%s/logs/"+str(file)+".csv"
    # natural_adam_mean = load_from_path(fpath, seeds=seeds)
    #
    # fpath = "results/"+str(tag)+"/"+str(env)+"/natural_amsgrad/optim_adaptive/shrunk_false/"+policy+"/total_samples_"+str(total_samples)+"/batch_size_"+str(bs)+"/lr_"+str(lr)+"/betas0.1_0.1/%s/logs/"+str(file)+".csv"
    # natural_amsgrad_mean = load_from_path(fpath, seeds=seeds)
    #
    # fpath = "results/"+str(tag)+"/"+str(env)+"/natural_adagrad/optim_adaptive/shrunk_false/"+policy+"/total_samples_"+str(total_samples)+"/batch_size_"+str(bs)+"/lr_"+str(lr)+"/%s/logs/"+str(file)+".csv"
    # natural_adagrad_mean = load_from_path(fpath, seeds=seeds)
    #
    # if show_shrunk:
    #     fpath = "results/"+str(tag)+"/"+str(env)+"/ngd/optim_adaptive/shrunk_true/lanczos_iters_"+str(lanczos_itr)+"/"+policy+"/total_samples_"+str(total_samples)+"/batch_size_"+str(bs)+"/lr_"+str(lr)+"/%s/logs/"+str(file)+".csv"
    #     ngd_cg_mean_shrunk = load_from_path(fpath, seeds=seeds)
    #
    #     fpath = "results/"+str(tag)+"/"+str(env)+"/natural_adam/optim_adaptive/shrunk_true/lanczos_iters_"+str(lanczos_itr)+"/"+policy+"/total_samples_"+str(total_samples)+"/batch_size_"+str(bs)+"/lr_"+str(lr)+"/betas0.1_0.1/%s/logs/"+str(file)+".csv"
    #     natural_adam_mean_shrunk = load_from_path(fpath, seeds=seeds)
    #
    #     fpath = "results/"+str(tag)+"/"+str(env)+"/natural_amsgrad/optim_adaptive/shrunk_true/lanczos_iters_"+str(lanczos_itr)+"/"+policy+"/total_samples_"+str(total_samples)+"/batch_size_"+str(bs)+"/lr_"+str(lr)+"/betas0.1_0.1/%s/logs/"+str(file)+".csv"
    #     natural_amsgrad_mean_shrunk = load_from_path(fpath, seeds=seeds)
    #
    #     fpath = "results/"+str(tag)+"/"+str(env)+"/natural_adagrad/optim_adaptive/shrunk_true/lanczos_iters_"+str(lanczos_itr)+"/"+policy+"/total_samples_"+str(total_samples)+"/batch_size_"+str(bs)+"/lr_"+str(lr)+"/%s/logs/"+str(file)+".csv"
    #     natural_adagrad_mean_shrunk = load_from_path(fpath, seeds=seeds)


    fpath = "results/"+str(tag)+"/"+str(env)+"/trpo/ngd/optim_adaptive/shrunk_false/"+policy+"/total_samples_"+str(total_samples)+"/batch_size_"+str(bs)+"/lr_"+str(lr)+"/%s/logs/"+str(file)+".csv"
    trpo_mean, trpo_stdev = load_from_path(fpath, seeds=seeds, compile=compile)

    fpath = "results/"+str(tag)+"/"+str(env)+"/npg/ngd/optim_adaptive/shrunk_false/"+policy+"/total_samples_"+str(total_samples)+"/batch_size_"+str(bs)+"/lr_0.01/%s/logs/"+str(file)+".csv"
    npg_mean, npg_stdev = load_from_path(fpath, seeds=seeds, compile=compile)

    fpath = "results/"+str(tag)+"/"+str(env)+"/trpo/natural_adam/optim_adaptive/shrunk_false/"+policy+"/total_samples_"+str(total_samples)+"/batch_size_"+str(bs)+"/lr_"+str(lr)+"/betas0.1_0.1/%s/logs/"+str(file)+".csv"
    natural_adam_mean, _ = load_from_path(fpath, seeds=seeds, compile=compile)

    fpath = "results/"+str(tag)+"/"+str(env)+"/trpo/natural_amsgrad/optim_adaptive/shrunk_false/"+policy+"/total_samples_"+str(total_samples)+"/batch_size_"+str(bs)+"/lr_"+str(lr)+"/betas0.1_0.1/%s/logs/"+str(file)+".csv"
    natural_amsgrad_mean, _ = load_from_path(fpath, seeds=seeds, compile=compile)

    if show_shrunk:
        fpath = "results/"+str(tag)+"/"+str(env)+"/trpo/ngd/optim_adaptive/shrunk_true/lanczos_iters_"+str(lanczos_itr)+"/"+policy+"/total_samples_"+str(total_samples)+"/batch_size_"+str(bs)+"/lr_"+str(lr)+"/%s/logs/"+str(file)+".csv"
        trpo_mean_shrunk, _ = load_from_path(fpath, seeds=seeds, compile=compile)

        fpath = "results/"+str(tag)+"/"+str(env)+"/trpo/natural_adam/optim_adaptive/shrunk_true/lanczos_iters_"+str(lanczos_itr)+"/"+policy+"/total_samples_"+str(total_samples)+"/batch_size_"+str(bs)+"/lr_"+str(lr)+"/betas0.1_0.1/%s/logs/"+str(file)+".csv"
        natural_adam_shrunk_mean, natural_adam_shrunk_stdev = load_from_path(fpath, seeds=seeds, compile=compile)

        fpath = "results/"+str(tag)+"/"+str(env)+"/npg/natural_adam/optim_adaptive/shrunk_true/lanczos_iters_5/"+policy+"/total_samples_"+str(total_samples)+"/batch_size_"+str(bs)+"/lr_0.01/betas0.1_0.1/%s/logs/"+str(file)+".csv"
        npg_adam_shrunk_mean, npg_adam_shrunk_stdev = load_from_path(fpath, seeds=seeds, compile=compile)

        fpath = "results/"+str(tag)+"/"+str(env)+"/trpo/natural_amsgrad/optim_adaptive/shrunk_true/lanczos_iters_"+str(lanczos_itr)+"/"+policy+"/total_samples_"+str(total_samples)+"/batch_size_"+str(bs)+"/lr_"+str(lr)+"/betas0.1_0.1/%s/logs/"+str(file)+".csv"
        natural_amsgrad_mean_shrunk, _ = load_from_path(fpath, seeds=seeds, compile=compile)

    # plt.rc('font', family='serif')
    # plt.rc('text', usetex=True)
    #
    # SMALL_SIZE = 8
    # MEDIUM_SIZE = 10
    # BIGGER_SIZE = 12
    #
    # plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    # plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    # plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    # plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    # plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    # plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    # plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    #
    # fig = plt.figure(figsize=(4, 3))

    # lw = 1.0
    # plt.plot(natural_adam_mean, label="FANG-Adam$^*$", ls='solid', linewidth=lw, color='xkcd:light blue')
    # plt.plot(natural_amsgrad_mean, label="FANG-AMSGrad$^*$", ls='solid', linewidth=lw, color='xkcd:blue')
    # # plt.plot(natural_adagrad_mean, label="FANG-Adagrad$^*$", ls='solid', linewidth=lw, color='xkcd:dark blue')
    # plt.plot(ngd_cg_mean, label="NGD", ls='solid', linewidth=lw, color='xkcd:orange')
    # plt.plot(trpo_mean, label="TRPO", ls='solid', color='xkcd:green')
    #
    # if show_shrunk:
    #     plt.plot(natural_adam_mean_shrunk, label="FANG-Adam$^{*(s)}$", ls='dashed', linewidth=lw, color='xkcd:light blue')
    #     plt.plot(natural_amsgrad_mean_shrunk, label="FANG-AMSGrad$^{*(s)}$", ls='dashed', linewidth=lw, color='xkcd:blue')
    #     plt.plot(natural_adagrad_mean_shrunk, label="FANG-Adagrad$^{*(s)}$", ls='dashed', linewidth=lw, color='xkcd:dark blue')
    #     plt.plot(ngd_cg_mean_shrunk, label="NGD$^{(s)}$", ls='dashed', linewidth=lw, color='xkcd:orange')

    lw = 1.0
    # plt.plot(natural_adam_mean, label="FANG-Adam$^*$", ls='solid', linewidth=lw, color='xkcd:light blue')
    # plt.plot(natural_amsgrad_mean, label="FANG-AMSGrad$^*$", ls='solid', linewidth=lw, color='xkcd:blue')
    # plt.plot(natural_adagrad_mean, label="FANG-Adagrad$^*$", ls='solid', linewidth=lw, color='xkcd:dark blue')

    if show_shrunk:
        ax.plot(natural_adam_shrunk_mean, label="AdaCurv-Adam$^{*(s)}$-TRPO", ls='solid', linewidth=lw, color='xkcd:fuchsia')
        # ax.fill_between(np.arange(len(natural_adam_shrunk_mean)), natural_adam_shrunk_mean-natural_adam_shrunk_stdev, natural_adam_shrunk_mean+natural_adam_shrunk_stdev, alpha=0.1, color='xkcd:fuchsia', zorder=1000)

        # ax.plot(npg_adam_shrunk_mean, label="FANG-Adam$^{*(s)}$-NPG", ls='solid', linewidth=lw, color='xkcd:orange')
        # ax.fill_between(np.arange(len(npg_adam_shrunk_mean)), npg_adam_shrunk_mean-npg_adam_shrunk_stdev, npg_adam_shrunk_mean+npg_adam_shrunk_stdev, alpha=0.1, color='xkcd:orange', zorder=1000)

        # plt.plot(natural_amsgrad_mean_shrunk, label="FANG-AMSGrad$^{*(s)}$", ls='dashed', linewidth=lw, color='xkcd:blue')
        # plt.plot(natural_adagrad_mean_shrunk, label="FANG-Adagrad$^{*(s)}$", ls='dashed', linewidth=lw, color='xkcd:dark blue')
        # plt.plot(trpo_mean_shrunk, label="TRPO$^{(s)}$", ls='dashed', linewidth=lw, color='xkcd:orange')

    ax.plot(trpo_mean, label="TRPO", ls='solid', linewidth=lw, color='#2F4F4F')
    # ax.fill_between(np.arange(len(trpo_mean)), trpo_mean-trpo_stdev, trpo_mean+trpo_stdev, alpha=0.1, color='#2F4F4F')

    # ax.plot(npg_mean, label="NPG", ls='solid', linewidth=lw, color='#778899')
    # ax.fill_between(np.arange(len(npg_mean)), npg_mean-npg_stdev, npg_mean+npg_stdev, alpha=0.1, color='#778899')


    if env == 'HopperBulletEnv-v0':
        ax.set_ylabel("Reward")
    ax.set_xlabel(env[:-12]) #"Batch " + str(bs))

    ax.xaxis.set_major_locator(ticker.MultipleLocator(50.0))
    plt.setp(ax.yaxis.get_majorticklabels(), rotation=-45)
    ax.tick_params(axis='y', which='major', pad=-5)


    # ax.xticks(rotation=45)
    # plt.ylabel("Reward")
    # plt.xlabel("Iteration")
    # plt.tight_layout()
    # plt.legend()

    sns.despine()

    # plt.savefig("results/"+str(tag)+"/plots/"+ subtag +"/"+str(env)+"_bs"+str(bs)+"_rewards_"+compile+".pdf")

f, axes = plt.subplots(1, 1)
tag = 'results_1mil/pybullet_sample_mode_trpo'
subtag = 'quick_start'
plot(tag=tag, env='HopperBulletEnv-v0', subtag=subtag, compile='mean', fig_ax=(f, axes))
# plot(tag=tag, env='HalfCheetahBulletEnv-v0', subtag=subtag, compile='mean', fig_ax=(f, axes[1]))
# plot(tag=tag, env='Walker2DBulletEnv-v0', subtag=subtag, compile='mean', fig_ax=(f, axes[2]))


f.set_figwidth(4)
f.set_figheight(3)

# plt.subplots_adjust(wspace=.25, hspace=0.0)

# Legends: https://stackoverflow.com/questions/4700614/how-to-put-the-legend-out-of-the-plot/27355247#27355247
plt.legend()
# plt.legend(loc='upper center', bbox_to_anchor=(-0.8, 1.4),
#           ncol=2, fancybox=False, shadow=False)

# plt.legend(loc='upper center', bbox_to_anchor=(1, 0.5))
plt.savefig("results/"+str(tag)+"/plots/"+ subtag +"/full.pdf", bbox_inches='tight')


# plot(tag='pybullet_sample_mode_full', env='HopperBulletEnv-v0', subtag='final', compile='mean')
# plot(tag='pybullet_sample_mode_full', env='HalfCheetahBulletEnv-v0', subtag='final', compile='mean')
# plot(tag='pybullet_sample_mode_full', env='Walker2DBulletEnv-v0', subtag='final', compile='mean')
