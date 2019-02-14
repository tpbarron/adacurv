import copy
import arguments
from itertools import product, chain
import ray
import mnist

ray.init(num_cpus=4)


###
# Common params
###

seeds = list(range(1))
global_lrs = [0.001]
batch_sizes = [125, 250, 500, 1000]
decay = True
epochs = 10
verbose = False

###
# Ray remote function simply to launch the run
###

@ray.remote
def run(args):
    print ("Starting job with args: ", args)
    mnist.launch_job(args)
    print ("Finished job with args: ", args)

###
# Start a set of variants
###

def run_variants(variants):
    gets = []

    for variant in variants:
        seed, optim, curv_type, lr, bs, cg_iters, cg_prev_init_coef, cg_precondition_empirical, cg_precondition_regu_coef, cg_precondition_exp, shrinkage_method, lanczos_amortization, lanczos_iters, bts, approx_adaptive = variant

        args = arguments.get_args()
        args.optim = optim
        args.curv_type = curv_type
        args.lr = lr
        args.decay_lr = decay
        args.epochs = epochs
        args.batch_size = bs

        args.cg_iters = cg_iters
        args.cg_prev_init_coef = cg_prev_init_coef
        args.cg_precondition_empirical = cg_precondition_empirical
        args.cg_precondition_regu_coef = cg_precondition_regu_coef
        args.cg_precondition_exp = cg_precondition_exp

        args.shrinkage_method = shrinkage_method
        args.lanczos_amortization = lanczos_amortization
        args.lanczos_iters = lanczos_iters

        args.beta1 = bts[0]
        args.beta2 = bts[1]
        args.approx_adaptive = approx_adaptive

        args.seed = seed
        args.log_dir = 'results/'+str(tag)
        args.verbose = verbose

        pid = run.remote(args)
        gets.append(pid)

    ray.get([pid for pid in gets])

###
# Run baselines only once
###

tag = 'baselines'
lrs = [0.1, 0.05, 0.001]
variants1 = product(seeds,
                    ['sgd', 'adam', 'amsgrad', 'adagrad'],      # optim
                    [''],                                       # curv_type
                    lrs,                                        # lr
                    batch_sizes,                                # batch size
                    [10],                                       # cg_iters
                    [0.0],                                      # cg_prev_init_coef
                    [False],                                    # cg_precondition_empirical
                    [0.0],                                      # cg_precondition_regu_coef
                    [0.0],                                      # cg_precondition_exp
                    [None],                                     # shrinkage_method
                    [0],                                        # lanzcos_amortization
                    [0],                                        # lanzcos_iters
                    [(0.0, 0.0)],                               # betas (ignored for these optimizers)
                    [False])                                    # approx adaptive

variants1 = copy.deepcopy(list(variants1))
print (variants1)
print (len(variants1))
input("Continue?")
run_variants(variants1)


###
# Run all basic with no preconditioner, no momentum, no amortization, to ensure code not broken
###

###
# Run only changing to gauss_newton to observe if significant difference
###

###
# Test change in shrinkage
###

###
# Test preconditioner
###

###
# Test momentum
###

# tag = 'mnist_basic'
#
# seeds = list(range(5))
# algos1 = ['ngd', 'natural_adagrad']
# algos2 = ['natural_adam', 'natural_amsgrad']
# shrunk_ks = [10]
# batch_sizes = [125, 250, 500, 1000]
# betas = [(0.1, 0.1)]
# lrs = [0.001]
# decay = True
# verbose = False
# epochs = 10
#
# # seed x optim x shrunk(bool), lanczos k x batch size x lr x approx adaptive x betas
# variants1 = product(seeds, ['sgd', 'adam', 'amsgrad', 'adagrad'], [False], [0], batch_sizes, lrs, [False], [(0.0, 0.0)])
# # ngd versions without shrinkage, (both approx and optimal)
# approx_adaptive = [True, False]
# variants2a = product(seeds, ['ngd'], [False], shrunk_ks, batch_sizes, lrs, [False], betas)
# variants2b = product(seeds, ['natural_adagrad'], [False], shrunk_ks, batch_sizes, lrs, approx_adaptive, betas)
# variants2c = product(seeds, algos2, [False], shrunk_ks, [125], [0.0001], approx_adaptive, betas)
# variants2d = product(seeds, algos2, [False], shrunk_ks, [250], [0.0005], approx_adaptive, betas)
# variants2e = product(seeds, algos2, [False], shrunk_ks, [500], [0.001], approx_adaptive, betas)
# variants2f = product(seeds, algos2, [False], shrunk_ks, [1000], [0.001], approx_adaptive, betas)
# variants2 = list(chain(variants2a, variants2b, variants2c, variants2d, variants2e, variants2f))
#
# approx_adaptive = [False]
# # ngd versions with shrinkage (only optimal)
# variants3a = product(seeds, algos1, [True], shrunk_ks, batch_sizes, lrs, approx_adaptive, betas)
# variants3b = product(seeds, algos2, [True], shrunk_ks, [125], [0.0001], approx_adaptive, betas)
# variants3c = product(seeds, algos2, [True], shrunk_ks, [250], [0.0005], approx_adaptive, betas)
# variants3d = product(seeds, algos2, [True], shrunk_ks, [500], [0.001], approx_adaptive, betas)
# variants3 = list(chain(variants3a, variants3b, variants3c, variants3d))
#
# all_variants = copy.deepcopy(list(chain(variants1, variants2, variants3)))
#
# print (list(all_variants))
# print (len(list(all_variants)))
# input("Continue?")
