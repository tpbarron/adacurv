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
                            True,                     # whether to use shrinkage
                            5,                        # num eigs to compute
                            5000,                     # batch size
                            0.0,                      # TRPO ignores this and uses KL dist 0.01
                            False,                    # whether to use approx adaptive update
                            (0.1, 0.1),               # betas for gradient and Fisher
                            True,                     # use neural net policy
                            1000000]                  # total samples
variant_trpo = [0, 'HopperBulletEnv-v0', 'trpo', 'ngd', False, 0, 5000, 0.0, False, (0.0, 0.0), True, 1000000]
variants = [variant_trpo_natural_adam]#, variant_trpo]

for s in seeds:
    for v in variants:
        v[0] = s # assign seed
        ray_train.launch_job(tag, v)
