import copy
import os
import time
from itertools import product, chain
import ray
import ray_train

ray.init()

tag = 'pybullet_sample_mode'
envs = ['Walker2DBulletEnv-v0', 'HalfCheetahBulletEnv-v0', 'HopperBulletEnv-v0']
seeds = [0, 1, 2]
algos = ['trpo', 'npg']
optims = ['natural_adam', 'natural_amsgrad']
shrunk_ks = [5] #167
batch_sizes = [5000]
approx_adaptive = [False]
betas = [(0.1, 0.1)]
lrs = [0.01] # NOTE: if have more than 1 lr, make sure TRPO only runs once!
use_nn_policy = [True]
total_samples = 1000000

# seed x envs x alg x optim x shrunk(bool), lanczos k x batch size x lr x approx adaptive x betas x (use nn policy)
# ngd versions without shrinkage
variants1a = product(seeds, envs, algos, ['ngd'], [False], [0], batch_sizes, lrs, [False], [(0.0, 0.0)], use_nn_policy, [total_samples])
variants1b = product(seeds, envs, algos, optims, [False], [0], batch_sizes, lrs, approx_adaptive, betas, use_nn_policy, [total_samples])
# ngd versions with shrinkage
variants2a = product(seeds, envs, algos, ['ngd'], [True], shrunk_ks, batch_sizes, lrs, [False], [(0.0, 0.0)], use_nn_policy, [total_samples])
variants2b = product(seeds, envs, algos, optims, [True], shrunk_ks, batch_sizes, lrs, approx_adaptive, betas, use_nn_policy, [total_samples])

all_variants = copy.deepcopy(list(chain(variants1a, variants1b, variants2a, variants2b)))
print (list(all_variants))
print (len(all_variants))
input("Continue?")

@ray.remote
def run(tag, variant):
    ts = time.time()
    print ("Starting job with args: ", variant)
    ray_train.launch_job(tag, variant)
    print ("Finished job with args: ", variant, " in ", (time.time()-ts))

gets = []
for variant in all_variants:
    pid = run.remote(tag, variant)
    gets.append(pid)

ray.get([pid for pid in gets])
