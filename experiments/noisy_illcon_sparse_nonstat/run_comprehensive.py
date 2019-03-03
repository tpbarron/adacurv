import copy
import arguments
from itertools import product, chain
import ray
import main

ray.init()

###
# Common params
###

seeds = list(range(1))
batch_sizes = [10]
iters = [500]
condition = [1.0, 100.0] #10.0, 100.0]
noise = [0.0] #0.1] #0.0, 0.4] #[0.0, 1.0, 5.0]
sparsity = [0.0, 0.1] #, 0.5]
rotate = [False, True]
adaptive = [False, True]
nonstat = True#, True]

###
# Ray remote function simply to launch the run
###

@ray.remote
def run(args, i, n):
    print ("Starting job (" + str(i) + "/" + str(n) +") with args: ", args)
    main.launch_job(args)
    print ("Finished job (" + str(i) + "/" + str(n) + ") with args: ", args)

###
# Start a set of variants
###

def run_variants(variants):
    gets = []

    i = 1
    n = len(variants)
    for variant in variants:
        tag, seed, bs, iters, cond, noise, sparsity, rotate, adaptive = variant

        args = arguments.get_args()
        args.batch_size = bs
        args.iters = iters
        args.dimension = 100
        args.condition = cond
        args.noise = noise
        args.grad_sparsity = sparsity
        args.rotate = rotate
        args.adaptive = adaptive
        args.nonstat = nonstat

        args.seed = seed
        args.log_dir = 'results/'+str(tag)

        pid = run.remote(args, i, n)
        i += 1
        gets.append(pid)

    ray.get([pid for pid in gets])


all_variants = []
tag = 'quadratic_comp'
variants1 = product([tag],
                    seeds,
                    batch_sizes,                                # batch size
                    iters,
                    condition,
                    noise,
                    sparsity,
                    rotate,
                    adaptive)

variants1 = copy.deepcopy(list(variants1))
print (len(variants1))
all_variants = copy.deepcopy(list(chain(all_variants, variants1)))

print ("Total:", len(all_variants))
input("Continue?")
run_variants(all_variants)
