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
        # maybe_fix_data(seeded_path)
        data_seed = np.load(seeded_path)
        # print (data_seed)
        ngd_cg_seeds.append(data_seed[np.newaxis,:])
    # input("")
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

def plot(tag='mlp', epochs=10, bs=250, subtag='batch', lr='0.001', file='data', fig_ax=None, seeds=[0, 1, 2]):
    try:
        os.makedirs("results/"+str(tag)+"/plots/" + subtag)
    except:
        pass

    if fig_ax is not None:
        fig, ax = fig_ax

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

    fname = "results/"+str(tag)+"/adagrad/optim_adaptive/shrunk_false/batch_size_"+str(bs)+"/lr_"+str(lr)+"/%s/data.npy"
    (adagrad, adagrad_std), (adagrad_max_mean, adagrad_max_std) = load_from_path(fname, seeds=seeds)
    print ("adagrad: ", adagrad_max_mean, "+/-", adagrad_max_std)

    fname = "results/"+str(tag)+"/amsgrad/optim_adaptive/shrunk_false/batch_size_"+str(bs)+"/lr_"+str(lr)+"/%s/data.npy"
    (amsgrad, amsgrad_std), (amsgrad_max_mean, amsgrad_max_std) = load_from_path(fname, seeds=seeds)
    print ("amsgrad: ", amsgrad_max_mean, "+/-", amsgrad_max_std)

    fname = "results/"+str(tag)+"/sgd/optim_adaptive/shrunk_false/batch_size_"+str(bs)+"/lr_"+str(lr)+"/%s/data.npy"
    (sgd, sgd_std), (sgd_max_mean, sgd_max_std) = load_from_path(fname, seeds=seeds)
    print ("sgd: ", sgd_max_mean, "+/-", sgd_max_std)

    fname = "results/"+str(tag)+"/ngd/optim_adaptive/shrunk_false/batch_size_"+str(bs)+"/lr_"+str(lr)+"/%s/data.npy"
    (ngd, ngd_std), (ngd_max_mean, ngd_max_std) = load_from_path(fname, seeds=seeds)
    print ("ngd: ", ngd_max_mean, "+/-", ngd_max_std)

    fname = "results/"+str(tag)+"/natural_adagrad/optim_adaptive/shrunk_false/batch_size_"+str(bs)+"/lr_"+str(lr)+"/%s/data.npy"
    (natural_adagrad_opt, natural_adagrad_opt_std), (natural_adagrad_max_mean, natural_adagrad_max_std) = load_from_path(fname, seeds=seeds)
    print ("natural_adagrad_opt: ", natural_adagrad_max_mean, "+/-", natural_adagrad_max_std)

    fname = "results/"+str(tag)+"/natural_adagrad/approx_adaptive/shrunk_false/batch_size_"+str(bs)+"/lr_"+str(lr)+"/%s/data.npy"
    (natural_adagrad_opt, natural_adagrad_opt_std), (natural_adagrad_max_mean, natural_adagrad_max_std) = load_from_path(fname, seeds=seeds)
    print ("natural_adagrad_approx: ", natural_adagrad_max_mean, "+/-", natural_adagrad_max_std)

    # fname = "results/"+str(tag)+"/natural_adagrad/optim_adaptive/shrunk_true/lanczos_iters_10/batch_size_"+str(bs)+"/lr_"+str(lr)+"/%s/data.npy"
    # (natural_adagrad_opt, natural_adagrad_opt_std), (natural_adagrad_max_mean, natural_adagrad_max_std) = load_from_path(fname, seeds=seeds)
    # print ("natural_adagrad_opt_shrunk: ", natural_adagrad_max_mean, "+/-", natural_adagrad_max_std)

    # natural_adagrad_approx = np.load("results/"+str(tag)+"/natural_adagrad/approx_adaptive/shrunk_false/batch_size_"+str(bs)+"/lr_"+str(lr)+"/"+str(seed)+"/"+str(file)+".npy")
    # natural_adagrad_opt_shrunk = np.load("results/"+str(tag)+"/natural_adagrad/optim_adaptive/shrunk_true/lanczos_iters_10/batch_size_"+str(bs)+"/lr_"+str(lr)+"/0/"+str(file)+".npy")

    # Reduce LR for nat adam and nat amsgrad
    if bs in [125]:
        lr = 0.0001
    elif bs in [250]:
        lr = 0.0005

    fname = "results/"+str(tag)+"/natural_amsgrad/optim_adaptive/shrunk_false/batch_size_"+str(bs)+"/lr_"+str(lr)+"/betas0.1_0.1/%s/data.npy"
    (natural_amsgrad_opt, natural_amsgrad_opt_std), (natural_amsgrad_max_mean, natural_amsgrad_max_std) = load_from_path(fname, seeds=seeds)
    print ("natural_amsgrad_opt: ", natural_amsgrad_max_mean, "+/-", natural_amsgrad_max_std)
    fname = "results/"+str(tag)+"/natural_amsgrad/approx_adaptive/shrunk_false/batch_size_"+str(bs)+"/lr_"+str(lr)+"/betas0.1_0.1/%s/data.npy"
    (natural_amsgrad_opt, natural_amsgrad_opt_std), (natural_amsgrad_max_mean, natural_amsgrad_max_std) = load_from_path(fname, seeds=seeds)
    print ("natural_amsgrad_approx: ", natural_amsgrad_max_mean, "+/-", natural_amsgrad_max_std)

    # natural_amsgrad_approx = np.load("results/"+str(tag)+"/natural_amsgrad/approx_adaptive/shrunk_false/batch_size_"+str(bs)+"/lr_"+str(lr)+"/betas0.1_0.1/"+str(seed)+"/"+str(file)+".npy")
    # natural_amsgrad_opt_shrunk = np.load("results/"+str(tag)+"/natural_amsgrad/optim_adaptive/shrunk_true/lanczos_iters_10/batch_size_"+str(bs)+"/lr_"+str(lr)+"/betas0.1_0.1/0/"+str(file)+".npy")

    fname = "results/"+str(tag)+"/natural_adam/optim_adaptive/shrunk_false/batch_size_"+str(bs)+"/lr_"+str(lr)+"/betas0.1_0.1/%s/data.npy"
    (natural_adam_opt, natural_adam_opt_std), (natural_adam_max_mean, natural_adam_max_std) = load_from_path(fname, seeds=seeds)
    print ("natural_adam_opt: ", natural_adam_max_mean, "+/-", natural_adam_max_std)

    fname = "results/"+str(tag)+"/natural_adam/approx_adaptive/shrunk_false/batch_size_"+str(bs)+"/lr_"+str(lr)+"/betas0.1_0.1/%s/data.npy"
    (natural_adam_opt, natural_adam_opt_std), (natural_adam_max_mean, natural_adam_max_std) = load_from_path(fname, seeds=seeds)
    print ("natural_adam_approx: ", natural_adam_max_mean, "+/-", natural_adam_max_std)
    # natural_adam_approx = np.load("results/"+str(tag)+"/natural_adam/approx_adaptive/shrunk_false/batch_size_"+str(bs)+"/lr_"+str(lr)+"/betas0.1_0.1/"+str(seed)+"/"+str(file)+".npy")
    # natural_adam_opt_shrunk = np.load("results/"+str(tag)+"/natural_adam/optim_adaptive/shrunk_true/lanczos_iters_10/batch_size_"+str(bs)+"/lr_"+str(lr)+"/betas0.1_0.1/0/"+str(file)+".npy")


    ##########
    print (np.round(natural_adam_max_mean, 2), "$\pm$", np.round(natural_adam_max_std, 2), "& ", end='')
    print (np.round(natural_amsgrad_max_mean, 2), "$\pm$", np.round(natural_amsgrad_max_std, 2), "& ", end='')
    print (np.round(natural_adagrad_max_mean, 2), "$\pm$", np.round(natural_adagrad_max_std, 2), "& ", end='')
    print (np.round(ngd_max_mean, 2), "$\pm$", np.round(ngd_max_std, 2), "& ", end='')
    print (np.round(adam_max_mean, 2), "$\pm$", np.round(adam_max_std, 2), "& ", end='')
    print (np.round(amsgrad_max_mean, 2), "$\pm$", np.round(amsgrad_max_std, 2), "& ", end='')
    print (np.round(adagrad_max_mean, 2), "$\pm$", np.round(adagrad_max_std, 2), "& ", end='')
    print (np.round(sgd_max_mean, 2), "$\pm$", np.round(sgd_max_std, 2))

    ##########
    # print ("Batch size: ", bs)
    # print ("Adam: ", max(adam))
    # print ("Adagrad: ", max(adagrad))
    # print ("AMSgrad: ", max(amsgrad))
    # print ("SGD: ", max(sgd))
    # print ("NGD: ", max(ngd))
    #
    # print ("natural_adagrad_approx:, ", max(natural_adagrad_approx))
    # print ("natural_adagrad_opt:, ", max(natural_adagrad_opt))
    # print ("natural_adagrad_opt_shrunk:, ", max(natural_adagrad_opt_shrunk))
    #
    # print ("natural_amsgrad_approx:, ", max(natural_amsgrad_approx))
    # print ("natural_amsgrad_opt:, ", max(natural_amsgrad_opt))
    # print ("natural_amsgrad_opt_shrunk:, ", max(natural_amsgrad_opt_shrunk))
    #
    # print ("natural_adam_approx:, ", max(natural_adam_approx))
    # print ("natural_adam_opt:, ", max(natural_adam_opt))
    # print ("natural_adam_opt_shrunk:, ", max(natural_adam_opt_shrunk))

    # fig = plt.figure(figsize=(4, 3))

    # # plt.plot(xs, natural_adam_approx, label="FANG-Adam-", ls='dashed', color='xkcd:light green')
    # # plt.plot(xs, natural_adam_opt, label="FANG-Adam$^*$", ls='solid', color='xkcd:dark green')
    # ax.plot(xs, natural_adam_opt, label="FANG-Adam$^*$", ls='solid', color='xkcd:fuchsia', zorder=1002, alpha=0.9)
    # ax.fill_between(xs, natural_adam_opt-natural_adam_opt_std, natural_adam_opt+natural_adam_opt_std, alpha=0.2, color='xkcd:fuchsia', zorder=1000)
    #
    # # print ("Best natural_adam_opt: ", max(natural_adam_opt))
    # # print ("Best natural_adam_approx: ", max(natural_adam_approx))
    #
    #
    # # plt.plot(xs, natural_amsgrad_approx, label="FANG-AMS-", ls='dashed', color='xkcd:light red')
    # # plt.plot(xs, natural_amsgrad_opt, label="FANG-AMSGrad$^*$", ls='solid', color='xkcd:dark red')
    # ax.plot(xs, natural_amsgrad_opt, label="FANG-AMSGrad$^*$", ls='solid', color='xkcd:orange', zorder=1001, alpha=0.9)
    # ax.fill_between(xs, natural_amsgrad_opt-natural_amsgrad_opt_std, natural_amsgrad_opt+natural_amsgrad_opt_std, alpha=0.2, color='xkcd:orange', zorder=1001)
    #
    # # print ("Best natural_amsgrad_opt: ", max(natural_amsgrad_opt))
    # # print ("Best natural_amsgrad_approx: ", max(natural_amsgrad_approx))
    #
    # # plt.plot(xs, natural_adagrad_approx, label="FANG-Ada-", ls='dashed', color='xkcd:light blue')
    # # plt.plot(xs, natural_adagrad_opt, label="FANG-Adagrad$^*$", ls='solid', color='xkcd:dark blue')
    # ax.plot(xs, natural_adagrad_opt, label="FANG-Adagrad$^*$", ls='solid', color='xkcd:blue', zorder=1000, alpha=0.9)
    # ax.fill_between(xs, natural_adagrad_opt-natural_adagrad_opt_std, natural_adagrad_opt+natural_adagrad_opt_std, alpha=0.2, color='xkcd:blue', zorder=1002)
    #
    # # print ("Best natural_adagrad_opt: ", max(natural_adagrad_opt))
    # # print ("Best natural_adagrad_approx: ", max(natural_adagrad_approx))
    #
    #
    # ax.plot(xs, ngd, label="NGD", ls='solid', color='#2F4F4F', alpha=0.9) #, marker="v", markevery=10, markersize=4) #color='xkcd:orange')
    # ax.fill_between(xs, ngd-ngd_std, ngd+ngd_std, alpha=0.2, color='#2F4F4F') #, color='xkcd:orange', alpha=0.2)
    #
    # ax.plot(xs, amsgrad, label="AMSGrad", ls='solid', color='#778899', alpha=0.9) #, color='xkcd:red')
    # ax.fill_between(xs, amsgrad-amsgrad_std, amsgrad+amsgrad_std, alpha=0.2, color='#778899') # color='xkcd:red', alpha=0.2)
    #
    # ax.plot(xs, adam, label="Adam", ls='dotted', color='#696969', alpha=0.9) #, marker=".", markevery=10, markersize=3) #, color='xkcd:green')
    # ax.fill_between(xs, adam-adam_std, adam+adam_std, alpha=0.2, color='#696969') #color='xkcd:green', alpha=0.2)
    #
    # ax.plot(xs, adagrad, label="Adagrad", ls='solid', color='#A9A9A9', alpha=0.9) #color='xkcd:blue')
    # ax.fill_between(xs, adagrad-adagrad_std, adagrad+adagrad_std, alpha=0.2, color='#A9A9A9') #color='xkcd:blue', alpha=0.2)
    #
    # ax.plot(xs, sgd, label="SGD", ls='solid', color='#D3D3D3', alpha=0.9) #, color='xkcd:teal')
    # ax.fill_between(xs, sgd-sgd_std, sgd+sgd_std, alpha=0.2, color='#D3D3D3') # color='xkcd:teal', alpha=0.2)
    #
    # # sns.set_palette("Blues")
    # # ax.plot(xs, ngd, label="NGD", ls='solid') #, color='#999999') #color='xkcd:orange')
    # # ax.fill_between(xs, ngd-ngd_std, ngd+ngd_std, alpha=0.2) #, color='#999999') #, color='xkcd:orange', alpha=0.2)
    # #
    # # ax.plot(xs, amsgrad, label="AMSGrad", ls='solid') #, color='#777777') #, color='xkcd:red')
    # # ax.fill_between(xs, amsgrad-amsgrad_std, amsgrad+amsgrad_std, alpha=0.2) #, color='#777777') # color='xkcd:red', alpha=0.2)
    # #
    # # ax.plot(xs, adam, label="Adam", ls='dashed') #, color='#555555') #, color='xkcd:green')
    # # ax.fill_between(xs, adam-adam_std, adam+adam_std, alpha=0.2) #, color='#555555') #color='xkcd:green', alpha=0.2)
    # #
    # # ax.plot(xs, adagrad, label="Adagrad", ls='solid') #, color='#333333') #color='xkcd:blue')
    # # ax.fill_between(xs, adagrad-adagrad_std, adagrad+adagrad_std, alpha=0.2) #, color='#333333') #color='xkcd:blue', alpha=0.2)
    # #
    # # ax.plot(xs, sgd, label="SGD", ls='solid') #, color='#111111') #, color='xkcd:teal')
    # # ax.fill_between(xs, sgd-sgd_std, sgd+sgd_std, alpha=0.2) #, color='#111111') # color='xkcd:teal', alpha=0.2)

    ylims=(90.0, 98.5)
    plt.ylim(ylims)
    # xlims=(0.0, 100)
    # plt.xlim(xlims)
    if bs == 125:
        ax.set_ylabel("Accuracy")

    ax.set_xlabel("Batch " + str(bs))

    ax.xaxis.set_major_locator(ticker.MultipleLocator(2.0))

    sns.despine()


epochs = 10
batch_sizes = [1000] #125, 250, 500, 1000]
f, axes = plt.subplots(1, 4, sharey=True)
tag='full' #mlp_mnist_seeds_decaysqrt'
subtag='test'

for b, ax in zip(batch_sizes, axes):
    # plot(tag='mlp_lanczos_fix_test', bs=b)
    # plot(tag='mlp_linesearch_test', bs=b)
    # plot(tag='mlp_highbeta_lowlr_test', bs=b)
    # plot(tag='mlp_highbeta_decaylr', bs=b)
    # plot(tag='mlp_mnist_batch_test', bs=b, subtag='adashrunk')
    # plot(tag='mlp_mnist_batch_test_biasfix_bfix_betas0.9_0.99', bs=b, subtag='batch')
    # plot(tag='mlp_mnist_batch_test_biasfix_bfix_cgpriorfix_betas0.9_0.99', bs=b, subtag='batch')
    # plot(tag='mlp_mnist_batch_test_biasfix_bfix_cgpriorfix_betas0.9_0.5', bs=b, subtag='batch')
    # plot(tag='mlp_mnist_batch_test_biasfix_bfix_cgpriorfix_betas0.1_0.9', bs=b, subtag='batch')
    # plot(tag='mlp_mnist_batch_test_biasfix_bfix_cgpriorfix_betas0.1_0.5', bs=b, subtag='batch')
    # plot(tag='mlp_mnist_batch_test_biasfix_bfix_cgpriorfix_betas0.1_0.1', bs=b, subtag='batch')
    # plot(tag='mlp_mnist_batch_test_biasfix_revertbfix_cgpriorfix_betas0.1_0.1', bs=b, subtag='batch')
    # plot(tag='mlp_mnist_batch_test_biasfix_bfix_revertcgpriorfix_betas0.1_0.1', bs=b, subtag='batch')
    # plot(tag='mlp_mnist_batch_test_biasfix_bfix_revertcgpriorfixghat_betas0.1_0.1', bs=b, subtag='batch')
    # plot(tag='mlp_mnist_batch_test_biasfix_bfix_revertcgpriorfixghat_betas0.5_0.5', bs=b, subtag='batch')
    # plot(tag='mlp_mnist_batch_test_biasfix_bfix_revertcgpriorfixghat_betas0.9_0.9', bs=b, subtag='batch')
    # plot(tag='mlp_mnist_batch_test_biasfix_bfix_cgpriorfixghat_peqn_betas0.9_0.9', bs=b, subtag='batch')
    # plot(tag='mlp_mnist_batch_test_biasfix_bfix_cgpriorfix_betas0.1_0.1_r2', bs=b, subtag='batch')
    # plot(tag='mlp_mnist_batch_test_biasfix_bfix_cgprior0_betas0.1_0.1', bs=b, subtag='batch')
    # plot(tag='mlp_mnist_batch_test_biasfix_bfix_cgprior0_betas0.5_0.5', bs=b, subtag='batch')
    # plot(tag='mlp_mnist_batch_test_biasfix_bfix_cgprior0_betas0.9_0.9', bs=b, subtag='batch')
    # plot(tag='mlp_mnist_batch_test_biasfix_bfix_cgprior0_betas0.9_0.9_lrdecay', bs=b, subtag='batch')
    # plot(tag='mlp_mnist_batch_test_biasfix_bfix_cgprior0_betas0.9_0.9_lrdecay_swapcgline', bs=b, subtag='batch')
    # plot(tag='linear_mnist_batch_test_biasfix_bfix_cgprior0_betas0.9_0.9_lrdecay_swapcgline', bs=b, subtag='batch')
    # plot(tag='mlp_mnist_batch_test_epochs2_lr0.001', bs=b, subtag='batch')
    # plot(tag='mlp_mnist_batch_test_epochs10_lr0.001', bs=b, subtag='batch')

    # plot(tag='mlp_mnist_baseline_epochs10_lr0.001', bs=b, subtag='batch')

    plot(tag=tag, epochs=10, bs=b, subtag='test', seeds=[0], fig_ax=(f, ax))

f.set_figwidth(8)
f.set_figheight(1.5)

# Legends: https://stackoverflow.com/questions/4700614/how-to-put-the-legend-out-of-the-plot/27355247#27355247
# plt.legend()
# plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
#           ncol=3, fancybox=True, shadow=True)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))


# plt.savefig("results/"+str(tag)+"/plots/"+ subtag +"/full.pdf", bbox_inches='tight')
