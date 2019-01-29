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

def plot(tag='mlp', bs=250, subtag='batch', lr='0.001', file='data'):
    print ("Tag, bs: ", tag, bs)
    try:
        os.makedirs("results/"+str(tag)+"/plots/" + subtag)
    except:
        pass

    epoch_ids = np.load("results/meta/epoch_ids_batch"+str(bs)+"_reduce.npy")
    # subsamp = gen_subsample(epoch_ids)
    # print (subsamp)
    xs = gen_xs(epoch_ids) #[subsamp]

    # epoch_ids_red = np.load("results/meta/epoch_ids_batch125_reduce.npy")
    # xs_red = gen_xs(epoch_ids_red)

    adam = np.load("results/"+str(tag)+"/adam/optim_adaptive/shrunk_false/batch_size_"+str(bs)+"/lr_"+str(lr)+"/0/"+str(file)+".npy")
    adagrad = np.load("results/"+str(tag)+"/adagrad/optim_adaptive/shrunk_false/batch_size_"+str(bs)+"/lr_"+str(lr)+"/0/"+str(file)+".npy")
    amsgrad = np.load("results/"+str(tag)+"/amsgrad/optim_adaptive/shrunk_false/batch_size_"+str(bs)+"/lr_"+str(lr)+"/0/"+str(file)+".npy")
    sgd = np.load("results/"+str(tag)+"/sgd/optim_adaptive/shrunk_false/batch_size_"+str(bs)+"/lr_"+str(lr)+"/0/"+str(file)+".npy")
    ngd = np.load("results/"+str(tag)+"/ngd/optim_adaptive/shrunk_false/batch_size_"+str(bs)+"/lr_"+str(lr)+"/0/"+str(file)+".npy")

    natural_adagrad_approx = np.load("results/"+str(tag)+"/natural_adagrad/approx_adaptive/shrunk_false/batch_size_"+str(bs)+"/lr_"+str(lr)+"/0/"+str(file)+".npy")
    natural_adagrad_opt = np.load("results/"+str(tag)+"/natural_adagrad/optim_adaptive/shrunk_false/batch_size_"+str(bs)+"/lr_"+str(lr)+"/0/"+str(file)+".npy")

    # Reduce LR for nat adam and nat amsgrad
    # if bs in [125]:
    #     lr = 0.0001
    # if bs in [250]:
    #     lr = 0.0005

    natural_adam_approx = np.load("results/"+str(tag)+"/natural_adam/approx_adaptive/shrunk_false/batch_size_"+str(bs)+"/lr_"+str(lr)+"/betas0.1_0.1/0/"+str(file)+".npy")
    natural_adam_opt = np.load("results/"+str(tag)+"/natural_adam/optim_adaptive/shrunk_false/batch_size_"+str(bs)+"/lr_"+str(lr)+"/betas0.1_0.1/0/"+str(file)+".npy")
    #
    natural_amsgrad_approx = np.load("results/"+str(tag)+"/natural_amsgrad/approx_adaptive/shrunk_false/batch_size_"+str(bs)+"/lr_"+str(lr)+"/betas0.1_0.1/0/"+str(file)+".npy")
    natural_amsgrad_opt = np.load("results/"+str(tag)+"/natural_amsgrad/optim_adaptive/shrunk_false/batch_size_"+str(bs)+"/lr_"+str(lr)+"/betas0.1_0.1/0/"+str(file)+".npy")

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

    plt.plot(xs, natural_adagrad_approx, label="FANG-Ada-", ls='dashed', color='xkcd:light blue')
    # plt.plot(xs, natural_adagrad_opt, label="FANG-Ada+", ls='solid', color='xkcd:dark blue')
    print ("Best natural_adagrad_opt: ", max(natural_adagrad_opt))
    print ("Best natural_adagrad_approx: ", max(natural_adagrad_approx))

    plt.plot(xs, natural_amsgrad_approx, label="FANG-AMS-", ls='dashed', color='xkcd:light red')
    # plt.plot(xs, natural_amsgrad_opt, label="FANG-AMS+", ls='solid', color='xkcd:dark red')
    print ("Best natural_amsgrad_opt: ", max(natural_amsgrad_opt))
    print ("Best natural_amsgrad_approx: ", max(natural_amsgrad_approx))

    plt.plot(xs, natural_adam_approx, label="FANG-Adam-", ls='dashed', color='xkcd:light green')
    # plt.plot(xs, natural_adam_opt, label="FANG-Adam+", ls='solid', color='xkcd:dark green')
    print ("Best natural_adam_opt: ", max(natural_adam_opt))
    print ("Best natural_adam_approx: ", max(natural_adam_approx))

    plt.plot(xs, ngd, label="NGD", ls='solid', color='xkcd:orange')
    print ("Best ngd: ", max(ngd))

    plt.plot(xs, amsgrad, label="AMSGrad", ls='solid', color='xkcd:red')
    plt.plot(xs, adam, label="Adam", ls='solid', color='xkcd:green')
    print ("Best Adam: ", max(adam))
    print ("Best AMSGrad: ", max(amsgrad))

    plt.plot(xs, adagrad, label="Adagrad", ls='solid', color='xkcd:blue')
    print ("Best adagrad: ", max(adagrad))
    plt.plot(xs, sgd, label="SGD", ls='solid', color='xkcd:teal')
    print ("Best sgd: ", max(sgd))

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

batch_sizes = [250, 500, 1000]
for b in batch_sizes:
    plot(tag='full_decayexp', bs=b, subtag='general')
