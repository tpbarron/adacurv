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

fig = plt.figure(figsize=(4, 3))

def load_from_path(fpath, seeds=[0], compile='mean', smooth=True):
    ngd_cg_seeds = []
    steps = None

    min_complete = np.inf
    for s in seeds:
        seeded_path = fpath % s
        print (seeded_path)
        with open(seeded_path, 'r') as f:
            reader = csv.reader(f)
            row1 = next(reader)  # gets the first line
            cind = row1.index('stoc_pol_mean')
            cind_steps = row1.index('steps')

            ngd_cg = np.loadtxt(seeded_path, delimiter=',', skiprows=2, usecols=(cind,))
            if len(ngd_cg) < min_complete:
                min_complete = len(ngd_cg)
            ngd_cg_seeds.append(ngd_cg[np.newaxis,:])
            print (ngd_cg.shape)

            steps = np.loadtxt(seeded_path, delimiter=',', skiprows=2, usecols=(cind_steps,))
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

    return steps, ngd_cg_mean, stdev

def plot(tag='mlp',
         subtag='batch',
         env='HalfCheetahBulletEnv-v0',
         lr='0.01',
         file='log',
         bs=5000,
         total_samples=2000000,
         policy='nn_policy',
         betas='betas0.1_0.1',
         lanczos_itr=5,
         seeds=[1, 2], #, 3, 4],
         show_shrunk=True,
         compile='mean',
         fig_ax=None):
    try:
        os.makedirs("results/"+str(tag)+"/plots/" + subtag)
    except:
        pass

    if fig_ax is not None:
        fig, ax = fig_ax

    fpath = "results/"+str(tag)+"/"+str(env)+"/trpo/ngd/optim_adaptive/curv_type_fisher/cg_iters_10/cg_residual_tol_1e-10/cg_prev_init_coef_0.5/cg_precondition_empirical_true/cg_precondition_regu_coef_0.001/cg_precondition_exp_0.75/shrunk_false/"+policy+"/adam_vfn_opt/total_samples_"+str(total_samples)+"/batch_size_"+str(bs)+"/lr_"+str(lr)+"/%s/logs/"+str(file)+".csv"
    trpo_steps, trpo_mean, trpo_stdev = load_from_path(fpath, seeds=seeds, compile=compile)

    fpath = "results/"+str(tag)+"/"+str(env)+"/trpo/natural_adam/optim_adaptive/curv_type_fisher/cg_iters_10/cg_residual_tol_1e-10/cg_prev_init_coef_0.5/cg_precondition_empirical_true/cg_precondition_regu_coef_0.001/cg_precondition_exp_0.75/shrunk_true/cg/"+policy+"/adam_vfn_opt/total_samples_"+str(total_samples)+"/batch_size_"+str(bs)+"/lr_"+str(lr)+"/"+betas+"/%s/logs/"+str(file)+".csv"
    nat_adam_steps, natural_adam_shrunk_mean, natural_adam_shrunk_stdev = load_from_path(fpath, seeds=seeds, compile=compile)

    lw = 1.0
    ls = 'dashed' if bs == 2000 else 'solid'
    print (len(nat_adam_steps), len(natural_adam_shrunk_mean))
    nat_adam_steps = nat_adam_steps[:len(natural_adam_shrunk_mean)]
    # nat_adam_steps = np.arange(len(natural_adam_shrunk_mean))
    plt.plot(nat_adam_steps, natural_adam_shrunk_mean, label=r"\noindent AdaCurv-Adam$^{*(s)}$-TRPO (batch: " + str(bs) + "; betas: " + betas[-3:] + ", " + betas[-3:] + ")", ls=ls, linewidth=lw, color='xkcd:fuchsia')
    plt.fill_between(nat_adam_steps, natural_adam_shrunk_mean-natural_adam_shrunk_stdev, natural_adam_shrunk_mean+natural_adam_shrunk_stdev, alpha=0.1, color='xkcd:fuchsia', zorder=1000)

    # trpo_steps = np.arange(len(trpo_mean))
    plt.plot(trpo_steps, trpo_mean, label="TRPO (batch: " + str(bs) + ")", ls=ls, linewidth=lw, color='#2F4F4F')
    plt.gca().fill_between(trpo_steps, trpo_mean-trpo_stdev, trpo_mean+trpo_stdev, alpha=0.1, color='#2F4F4F')


    # plt.savefig("results/"+str(tag)+"/plots/"+ subtag +"/"+str(env)+"_bs"+str(bs)+"_rewards_"+compile+".pdf")

# f, axes = plt.subplots(1, 3)
subtag = 'rerun2'
tag='pybullet_sample_mode_bball_random_hoop'
plot(tag='pybullet_sample_mode_bball_random_hoop', env='BasketBallEnvRandomHoop-v0', bs=2000, betas='betas0.9_0.9', subtag=subtag, compile='median') #, fig_ax=(f, axes[0]))
plot(tag='pybullet_sample_mode_bball_random_hoop', env='BasketBallEnvRandomHoop-v0', bs=5000, betas='betas0.1_0.1', subtag=subtag, compile='median') #, fig_ax=(f, axes[0]))


plt.ylabel("Reward")
plt.xlabel("Samples")
plt.tight_layout()
# plt.legend()
plt.xticks([0, 500000, 1000000, 1500000, 2000000])

sns.despine()
# plt.show()

# f.set_figwidth(5)
# f.set_figheight(1.25)
#
# plt.subplots_adjust(wspace=.25, hspace=0.0)
# # plt.tight_layout()
#
# # Legends: https://stackoverflow.com/questions/4700614/how-to-put-the-legend-out-of-the-plot/27355247#27355247
# # plt.legend()
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), #(0.45, 1.2),
          ncol=1, fancybox=False, shadow=False)
# plt.legend(loc='lower left', bbox_to_anchor=(1, 0.5))
# plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.savefig("results/"+str(tag)+"/plots/"+ subtag +"/full.pdf", bbox_inches='tight')


# plot(tag='pybullet_sample_mode_full', env='HopperBulletEnv-v0', subtag='final', compile='mean')
# plot(tag='pybullet_sample_mode_full', env='HalfCheetahBulletEnv-v0', subtag='final', compile='mean')
# plot(tag='pybullet_sample_mode_full', env='Walker2DBulletEnv-v0', subtag='final', compile='mean')
