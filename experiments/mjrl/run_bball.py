import copy
import os
import time
from itertools import product, chain
import ray
import ray_train

ray.init()

tag = 'pybullet_sample_mode_bball_random_hoop'
envs = ['BasketballEnvRandomHoop-v0']

seeds = [0, 1, 2]
algos = ['trpo']
optims = ['natural_adam'] #, 'natural_amsgrad']
curv_type = ['fisher']

shrunk_ks = [10]
batch_sizes = [1000, 2000, 5000]
approx_adaptive = [False]
betas = [(0.1, 0.1)] #, (0.9, 0.9)]
lrs = [0.01] # NOTE: if have more than 1 lr, make sure TRPO only runs once!
use_nn_policy = [True]
gn_vfn_opt = [False, True]
total_samples = 2000000

cg_iters = 10
cg_prev_init_coef = [0.5]
cg_precondition_empirical = True
cg_precondition_regu_coef = 0.001
cg_precondition_exp = 0.75
shrinkage_method = ['cg']
lanczos_amortization = 0
lanczos_iters = 0

ngd_noshrinkage_nocgplus = False
ngd_noshrinkage_yescgplus = True

adangd_noshrinkage_nocgplus = False
adangd_noshrinkage_yescgplus = False

ngd_yesshrinkage_yescgplus = False
adangd_yesshrinkage_yescgplus = True

all_variants = []

if ngd_noshrinkage_nocgplus:
    # ngd versions without shrinkage and without cg
    variants1 = product(seeds,                     # seed
                         envs,                      # envs
                         algos,                     # alg
                         ['ngd'],                   # optim
                         curv_type,                 # curv_type
                         lrs,
                         batch_sizes,

                         [10],                      # cg_iters
                         [1e-10],                   # cg_residual_tol
                         [0.0],                     # cg_prev_init_coef
                         [False],                   # cg_precondition_empirical
                         [0.0],                     # cg_precondition_regu_coef
                         [0.0],                     # cg_precondition_exp
                         [None],                    # shrinkage_method
                         [0],                       # lanzcos_amortization
                         [0],                       # lanzcos_iters

                         [False],                   # approx adaptive
                         [(0.0, 0.0)],              # betas
                         use_nn_policy,             # use nn
                         gn_vfn_opt,                # whether to optimizer val func with GN
                         [total_samples])           # total steps

    variants1 = copy.deepcopy(list(variants1))
    print (len(variants1))
    all_variants = copy.deepcopy(list(chain(all_variants, variants1)))

if ngd_noshrinkage_yescgplus:
    # ngd version no shrinkage but with cg enhancements
    variants1 = product(seeds,                     # seed
                         envs,                      # envs
                         algos,                     # alg
                         ['ngd'],                   # optim
                         curv_type,                 # curv_type
                         lrs,
                         batch_sizes,

                         [10],                      # cg_iters
                         [1e-10],                   # cg_residual_tol
                         [0.5],                     # cg_prev_init_coef
                         [True],                   # cg_precondition_empirical
                         [0.001],                     # cg_precondition_regu_coef
                         [0.75],                     # cg_precondition_exp
                         [None],                    # shrinkage_method
                         [0],                       # lanzcos_amortization
                         [0],                       # lanzcos_iters

                         [False],                   # approx adaptive
                         [(0.0, 0.0)],              # betas
                         use_nn_policy,             # use nn
                         [False],                # whether to optimizer val func with GN
                         [total_samples])           # total steps

    variants1 = copy.deepcopy(list(variants1))
    print (len(variants1))
    all_variants = copy.deepcopy(list(chain(all_variants, variants1)))

if adangd_noshrinkage_nocgplus:
    # adaptive ngd no shrinkage no cg
    variants1 = product(seeds,                     # seed
                         envs,                      # envs
                         algos,                     # alg
                         optims,                   # optim
                         curv_type,                 # curv_type
                         lrs,
                         batch_sizes,

                         [10],                      # cg_iters
                         [1e-10],                   # cg_residual_tol
                         [0.0],                     # cg_prev_init_coef
                         [False],                   # cg_precondition_empirical
                         [0.0],                     # cg_precondition_regu_coef
                         [0.0],                     # cg_precondition_exp
                         [None],                    # shrinkage_method
                         [0],                       # lanzcos_amortization
                         [0],                       # lanzcos_iters

                         [False],                   # approx adaptive
                         betas,              # betas
                         use_nn_policy,             # use nn
                         gn_vfn_opt,                # whether to optimizer val func with GN
                         [total_samples])           # total steps

    variants1 = copy.deepcopy(list(variants1))
    print (len(variants1))
    all_variants = copy.deepcopy(list(chain(all_variants, variants1)))

if adangd_noshrinkage_yescgplus:
    # adaptive ngd versions no shrinkage with cg enhancements
    variants1 = product(seeds,                     # seed
                         envs,                      # envs
                         algos,                     # alg
                         optims,                   # optim
                         curv_type,                 # curv_type
                         lrs,
                         batch_sizes,

                         [10],                      # cg_iters
                         [1e-10],                   # cg_residual_tol
                         [0.5],                     # cg_prev_init_coef
                         [True],                   # cg_precondition_empirical
                         [0.001],                     # cg_precondition_regu_coef
                         [0.75],                     # cg_precondition_exp
                         [None],                    # shrinkage_method
                         [0],                       # lanzcos_amortization
                         [0],                       # lanzcos_iters

                         [False],                   # approx adaptive
                         betas,              # betas
                         use_nn_policy,             # use nn
                         gn_vfn_opt,                # whether to optimizer val func with GN
                         [total_samples])           # total steps

    variants1 = copy.deepcopy(list(variants1))
    print (len(variants1))
    all_variants = copy.deepcopy(list(chain(all_variants, variants1)))

if ngd_yesshrinkage_yescgplus:
    # ngd versions with shrinkage and cg enhancements
    variants1 = product(seeds,                     # seed
                         envs,                      # envs
                         algos,                     # alg
                         ['ngd'],                   # optim
                         curv_type,                 # curv_type
                         lrs,
                         batch_sizes,

                         [10],                      # cg_iters
                         [1e-10],                   # cg_residual_tol
                         [0.5],                     # cg_prev_init_coef
                         [True],                   # cg_precondition_empirical
                         [0.001],                     # cg_precondition_regu_coef
                         [0.75],                     # cg_precondition_exp
                         ['cg'],                    # shrinkage_method
                         [0],                       # lanzcos_amortization
                         [0],                       # lanzcos_iters

                         [False],                   # approx adaptive
                         [(0.0, 0.0)],              # betas
                         use_nn_policy,             # use nn
                         gn_vfn_opt,                # whether to optimizer val func with GN
                         [total_samples])           # total steps

    variants1 = copy.deepcopy(list(variants1))
    print (len(variants1))
    all_variants = copy.deepcopy(list(chain(all_variants, variants1)))

if adangd_yesshrinkage_yescgplus:
    variants1 = product(seeds,                     # seed
                         envs,                      # envs
                         algos,                     # alg
                         optims,                   # optim
                         curv_type,                 # curv_type
                         lrs,
                         batch_sizes,
                         [10],                      # cg_iters
                         [1e-10],                   # cg_residual_tol
                         [0.5],                     # cg_prev_init_coef
                         [True],                   # cg_precondition_empirical
                         [0.001],                     # cg_precondition_regu_coef
                         [0.75],                     # cg_precondition_exp
                         ['cg'],                    # shrinkage_method
                         [0],                       # lanzcos_amortization
                         [0],                       # lanzcos_iters
                         [False],                   # approx adaptive
                         betas,              # betas
                         use_nn_policy,             # use nn
                         gn_vfn_opt,                # whether to optimizer val func with GN
                         [total_samples])           # total steps

    variants1 = copy.deepcopy(list(variants1))
    print (len(variants1))
    all_variants = copy.deepcopy(list(chain(all_variants, variants1)))

print (all_variants)
num_variants = len(all_variants)
print (num_variants)
input("Continue?")

@ray.remote
def run(tag, variant, i, n):
    ts = time.time()
    print ("Starting job (" + str(i) + "/" + str(n) + ") with args: ", variant)
    ray_train.launch_job(tag, variant)
    print ("Finished job (" + str(i) + "/" + str(n) + ") with args: ", variant, " in ", (time.time()-ts))

gets = []
for variant in reversed(all_variants):
    i = 1
    pid = run.remote(tag, variant, i, num_variants)
    gets.append(pid)
    i += 1

ray.get([pid for pid in gets])
