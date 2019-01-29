import os
import time
import ray_train

tag = 'pybullet/quick_start'
variant = [0,                       # seed
          'HopperBulletEnv-v0',     # environment
          'trpo',                   # algorithm (in this case, computes step size)
          'natural_adam',           # optimizer (computes step direction)
          True,                     # whether to use shrinkage
          5,                        # num eigs to compute
          5000,                     # batch size
          0.0,                      # TRPO ignores this and uses KL dist 0.01
          False,                    # use optimal adaptive update
          (0.1, 0.1),               # betas for gradient and Fisher
          True,                     # use neural net policy
          1000000]                  # total samples

ray_train.launch_job(tag, variant)
