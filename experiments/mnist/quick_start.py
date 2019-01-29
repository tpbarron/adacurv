import arguments
import mnist

verbose = True
epochs = 5
decay = True
variant = [0,               # seed
          'natural_adam',   # optimizer
          False,            # no shrinkage (uses constant damping of 0.0001)
          -1,               # Lanczos iterations if shrinkage was True
          500,              # batch size
          0.001,            # learning rate
          False,            # use the 'optimal' adaptive update, not the approximate one
          (0.1, 0.1)]       # betas for weighted updates of gradient and Fisher.

seed, optim, shrunk, k, bs, lr, approx_adap, bts = variant
args = arguments.get_args()
args.log_dir = 'results/mlp_mnist/quick_start/'
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

mnist.launch_job(args)

# Todo: plot performance, plot performance vs time.
