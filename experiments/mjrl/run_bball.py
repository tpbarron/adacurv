import copy
import os
import time
from itertools import product, chain
import ray
import ray_train

ray.init()

tag = 'pybullet_sample_mode_bball'
envs = ['BasketballEnv-v0']

seeds = [0, 1, 2]
algos = ['trpo']
optims = ['natural_adam'] #, 'natural_amsgrad']
curv_type = ['fisher']

shrunk_ks = [10]
batch_sizes = [5000]
approx_adaptive = [False]
betas = [(0.1, 0.1)]
lrs = [0.01] # NOTE: if have more than 1 lr, make sure TRPO only runs once!
use_nn_policy = [True]
total_samples = 2500000

cg_iters = 10
cg_prev_init_coef = [0.5]
cg_precondition_empirical = True
cg_precondition_regu_coef = 0.001
cg_precondition_exp = 0.75
shrinkage_method = ['cg']
lanczos_amortization = 0
lanczos_iters = 0


# ngd versions without shrinkage
variants1a = product(seeds,                     # seed
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
                     [total_samples])           # total steps

# ngd version no shrinkage but with cg enhancements
variants1b = product(seeds,                     # seed
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
                     [total_samples])           # total steps

# # adaptive ngd no shrinkage no cg
# variants1c = product(seeds,                     # seed
#                      envs,                      # envs
#                      algos,                     # alg
#                      optims,                   # optim
#                      curv_type,                 # curv_type
#                      lrs,
#                      batch_sizes,

#                      [10],                      # cg_iters
#                      [1e-10],                   # cg_residual_tol
#                      [0.0],                     # cg_prev_init_coef
#                      [False],                   # cg_precondition_empirical
#                      [0.0],                     # cg_precondition_regu_coef
#                      [0.0],                     # cg_precondition_exp
#                      [None],                    # shrinkage_method
#                      [0],                       # lanzcos_amortization
#                      [0],                       # lanzcos_iters

#                      [False],                   # approx adaptive
#                      [(0.0, 0.0)],              # betas
#                      use_nn_policy,             # use nn
#                      [total_samples])           # total steps

# # adaptive ngd versions no shrinkage with cg enhancements
# variants1d = product(seeds,                     # seed
#                      envs,                      # envs
#                      algos,                     # alg
#                      optims,                   # optim
#                      curv_type,                 # curv_type
#                      lrs,
#                      batch_sizes,

#                      [10],                      # cg_iters
#                      [1e-10],                   # cg_residual_tol
#                      [0.5],                     # cg_prev_init_coef
#                      [True],                   # cg_precondition_empirical
#                      [0.001],                     # cg_precondition_regu_coef
#                      [0.75],                     # cg_precondition_exp
#                      [None],                    # shrinkage_method
#                      [0],                       # lanzcos_amortization
#                      [0],                       # lanzcos_iters

#                      [False],                   # approx adaptive
#                      [(0.0, 0.0)],              # betas
#                      use_nn_policy,             # use nn
#                      [total_samples])           # total steps

# ngd versions with shrinkage and cg enhancements
variants2a = product(seeds,                     # seed
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
                     [total_samples])           # total steps

variants2b = product(seeds,                     # seed
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
                     [(0.0, 0.0)],              # betas
                     use_nn_policy,             # use nn
                     [total_samples])           # total steps

all_variants = copy.deepcopy(list(chain(variants1a, variants1b, variants2a, variants2b)))
print (list(all_variants))
num_variants = len(all_variants)
print (len(all_variants))
input("Continue?")

@ray.remote
def run(tag, variant, i, n):
    ts = time.time()
    print ("Starting job (" + str(i) + "/" + str(n) + ") with args: ", variant)
    ray_train.launch_job(tag, variant)
    print ("Finished job (" + str(i) + "/" + str(n) + ") with args: ", variant, " in ", (time.time()-ts))

gets = []
for variant in all_variants:
    i = 1
    pid = run.remote(tag, variant, i, num_variants)
    gets.append(pid)
    i += 1

ray.get([pid for pid in gets])
