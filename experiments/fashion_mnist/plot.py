import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

sns.set()
sns.set_style("white")
sns.set_context("paper")

def gen_xs(epoch_ids):
    unique, counts = np.unique(epoch_ids, return_counts=True)
    num_dict = dict(zip(unique, counts))
    xs = []
    for i in range(1, max(epoch_ids)+1):
        count = num_dict[i]
        xs_i = np.linspace(i-1, i, count+1)
        xs.extend(xs_i)
    return np.array(xs)

def gen_subsample(epoch_ids, n=10):
    unique, counts = np.unique(epoch_ids, return_counts=True)
    num_dict = dict(zip(unique, counts))
    subsamp = []
    prev_ind = 0
    for i in range(1, max(epoch_ids)+1):
        count = num_dict[i]
        # gen n ids from [prev_ind, prev_ind+count]
        samps = count // (n+1)
        if samps == 0:
            samps = 1
        subsamp_i = list(range(prev_ind, prev_ind+count, samps))
        print (subsamp_i, prev_ind, count)
        prev_ind = prev_ind+count
        subsamp.extend(subsamp_i)
    return np.array(subsamp)

def plot(tag='mlp', bs=250, subtag='batch', lr='0.0005', file='data'):
    print ("Tag, bs: ", tag, bs)
    try:
        os.makedirs("results/"+str(tag)+"/plots/" + subtag)
    except:
        pass

    # epoch_ids = np.load("results/meta/epoch_ids_batch"+str(bs)+"_reduce.npy")
    # xs = gen_xs(epoch_ids)

    # epoch_ids_red = np.load("results/meta/epoch_ids_batch125_reduce.npy")
    # xs_red = gen_xs(epoch_ids_red)

    adam = np.load("results/"+str(tag)+"/adam/optim_adaptive/curv_type_/cg_iters_10/cg_residual_tol_1e-10/cg_prev_init_coef_0.0/cg_precondition_empirical_false/shrunk_false/batch_size_"+str(bs)+"/lr_0.001/0/"+str(file)+".npy")
    ngd = np.load("results/"+str(tag)+"/ngd/optim_adaptive/curv_type_fisher/cg_iters_10/cg_residual_tol_1e-10/cg_prev_init_coef_0.0/cg_precondition_empirical_false/shrunk_false/batch_size_"+str(bs)+"/lr_"+str(lr)+"/0/"+str(file)+".npy")

    natural_adam_approx = np.load("results/"+str(tag)+"/natural_adam/optim_adaptive/curv_type_fisher/cg_iters_10/cg_residual_tol_1e-10/cg_prev_init_coef_0.1/cg_precondition_empirical_true/cg_precondition_regu_coef_0.001/cg_precondition_exp_0.75/shrunk_false/batch_size_250/lr_0.001/betas0.9_0.9/0/"+str(file)+".npy")

    kfac = np.load("results/"+str(tag)+"/kfac/batch_size_250/lr_"+str(lr)+"/1/"+str(file)+".npy")

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

    plt.plot(natural_adam_approx, label="FANG-Adam-", ls='dashed', color='xkcd:light green')
    print ("Best natural_adam_approx: ", max(natural_adam_approx))

    plt.plot(ngd, label="NGD", ls='solid', color='xkcd:orange')
    print ("Best ngd: ", max(ngd))

    plt.plot(adam, label="Adam", ls='solid', color='xkcd:green')
    print ("Best Adam: ", max(adam))

    plt.plot(kfac*100.0, label='K-FAC', ls='solid', color='xkcd:sky blue')
    print ("Best kfac: ", max(kfac))

    ylims=(50.0, 98.25)
    plt.ylim(ylims)
    # xlims=(0.0, 100)
    # plt.xlim(xlims)
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.tight_layout()
    plt.legend()

    sns.despine()

    plt.savefig("results/"+str(tag)+"/plots/"+ subtag +"/bs"+str(bs)+".pdf")

batch_sizes = [250] #, 500, 1000]
for b in batch_sizes:
    plot(tag='fashion_mnist_cnn_large', bs=b, subtag='general')
