import copy
import arguments
from itertools import product, chain
import ray
import svhn

ray.init()

tag = 'mlp_svhn'
seeds = list(range(1))
algos1 = ['ngd', 'natural_adagrad']
algos2 = ['natural_adam', 'natural_amsgrad']
shrunk_ks = [10]
batch_sizes = [125, 250, 500, 1000]
betas = [(0.1, 0.1)]
lrs = [0.001]
decay = True
verbose = False
epochs = 10

# seed x optim x shrunk(bool), lanczos k x batch size x lr x approx adaptive x betas
variants1 = product(seeds, ['sgd', 'adam', 'amsgrad', 'adagrad'], [False], [0], batch_sizes, lrs, [False], [(0.0, 0.0)])
# ngd versions without shrinkage, (both approx and optimal)
approx_adaptive = [False]
variants2a = product(seeds, ['ngd'], [False], shrunk_ks, batch_sizes, lrs, [False], betas)
variants2b = product(seeds, ['natural_adagrad'], [False], shrunk_ks, batch_sizes, lrs, approx_adaptive, betas)
variants2c = product(seeds, algos2, [False], shrunk_ks, [125], [0.0001], approx_adaptive, betas)
variants2d = product(seeds, algos2, [False], shrunk_ks, [250], [0.0005], approx_adaptive, betas)
variants2e = product(seeds, algos2, [False], shrunk_ks, [500], [0.001], approx_adaptive, betas)
variants2f = product(seeds, algos2, [False], shrunk_ks, [1000], [0.001], approx_adaptive, betas)
variants2 = list(chain(variants2a, variants2b, variants2c, variants2d, variants2e, variants2f))

all_variants = copy.deepcopy(list(chain(variants1, variants2)))
print (all_variants)
print (len(all_variants))
input("Continue?")

@ray.remote
def run(args):
    print ("Starting job with args: ", args)
    svhn.launch_job(args)
    print ("Finished job with args: ", args)

gets = []

for variant in all_variants:
    seed, optim, shrunk, k, bs, lr, approx_adap, bts = variant

    args = arguments.get_args()
    args.log_dir = 'results/'+str(tag)
    args.seed = seed
    args.optim = optim
    args.shrunk = shrunk
    args.lanczos_iters = k
    args.batch_size = bs
    args.lr = lr
    args.beta1 = bts[0]
    args.beta2 = bts[1]
    args.verbose = verbose
    args.epochs = epochs
    args.approx_adaptive = approx_adap
    args.decay_lr = decay


    pid = run.remote(args)
    gets.append(pid)

ray.get([pid for pid in gets])
