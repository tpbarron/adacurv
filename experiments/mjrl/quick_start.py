import os
import time
import ray_train

# NOTE: there is high variance and still some stochastisticity in these runs so one seed is not
# necessarily representative of relative performance. All plots in the paper use 5 seeds.
tag = 'quick_start'
seeds = [0]

variant_trpo_natural_adam = [0,                       # seed
                            'HopperBulletEnv-v0',     # environment
                            'trpo',                   # algorithm (in this case, computes step size)
                            'natural_adam',           # optimizer (computes step direction)
                            'fisher',                 # Curv type
                            0.0,                      # lr, ignored by TRPO, uses KL 0.1
                            5000,                     # batch size
                            10,                       # CG iters
                            1e-10,                    # CG res tolerance
                            0.0,                      # CG prev init coef
                            False,                    # CG prec empirical
                            0.0,                      # CG prec regu coef
                            0.0,                      # CG prec exponent
                            'lanczos',                # shrinkage method
                            1,                        # lanczos amortization
                            10,                       # lanczos iters
                            False,                    # whether to use approx adaptive update
                            (0.1, 0.1),               # betas for gradient and Fisher
                            True,                     # use neural net policy
                            False,                    # GN value func opt
                            1000000]                  # total samples

variant_trpo =             [0,                       # seed
                            'HopperBulletEnv-v0',     # environment
                            'trpo',                   # algorithm (in this case, computes step size)
                            'ngd',           # optimizer (computes step direction)
                            'fisher',                 # Curv type
                            0.0,                      # lr, ignored by TRPO, uses KL 0.1
                            5000,                     # batch size
                            10,                       # CG iters
                            1e-10,                    # CG res tolerance
                            0.0,                      # CG prev init coef
                            False,                    # CG prec empirical
                            0.0,                      # CG prec regu coef
                            0.0,                      # CG prec exponent
                            'lanczos',                # shrinkage method
                            1,                        # lanczos amortization
                            10,                       # lanczos iters
                            False,                    # whether to use approx adaptive update
                            (0.0, 0.0),               # betas for gradient and Fisher
                            True,                     # use neural net policy
                            False,                    # GN value func opt
                            1000000]                  # total samples
variants = [variant_trpo_natural_adam, variant_trpo]

for s in seeds:
    for v in variants:
        v[0] = s # assign seed
        ray_train.launch_job(tag, v)
