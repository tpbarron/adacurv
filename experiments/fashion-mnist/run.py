import copy
import arguments
from itertools import product, chain
import ray
import fashion_mnist

ray.init()

tag = 'mlp_fashionmnist_epochs8_lr0.001_decayexp'
seeds = list(range(1))
algos = ['natural_adam', 'natural_adagrad', 'natural_amsgrad']
shrunk_ks = [20]
batch_sizes = [500, 1000]
approx_adaptive = [True, False]
betas = [(0.1, 0.1)] #, (0.9, 0.99)]
# lrs = [0.001]
lrs = [0.001] #
decay = True
verbose = False
epochs = 5

# seed x optim x shrunk(bool), lanczos k x batch size x lr x approx adaptive x betas
variants1a = product(seeds, ['sgd', 'adam', 'amsgrad', 'adagrad'], [False], [0], batch_sizes, lrs, [False], [(0.0, 0.0)])
variants1b = product(seeds, ['ngd'], [False], [0], batch_sizes, lrs, [False], [(0.0, 0.0)])
# ngd versions without shrinkage
variants2 = product(seeds, algos, [False], [0], batch_sizes, lrs, approx_adaptive, betas)
# ngd versions with shrinkage
# variants3 = product(seeds, algos, [True], shrunk_ks, batch_sizes, lrs, approx_adaptive, betas)

all_variants = copy.deepcopy(list(chain(variants1a, variants1b, variants2)))
print (list(all_variants))
print (len(list(all_variants)))
input("Continue?")

@ray.remote
def run(args):
    print ("Starting job with args: ", args)
    fashion_mnist.launch_job(args)
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
