from mjrl.utils.gym_env import GymEnv
from mjrl.policies.gaussian_mlp import MLP
from mjrl.policies.gaussian_linear import LinearPolicy
from mjrl.baselines.quadratic_baseline import QuadraticBaseline
from mjrl.baselines.mlp_baseline import MLPBaseline
from mjrl.utils.train_agent import train_agent
from mjrl.algos.npg_cg import NPG

import abng.optim as fisher_optim
import mjrl.envs
import time as timer
SEED = 17

# NN policy
# ==================================
e = GymEnv('InvertedPendulumBulletEnv-v0')
# e = GymEnv('LunarLanderContinuous-v2')
# policy = MLP(e.spec, hidden_sizes=(32,32), seed=SEED)
policy = LinearPolicy(e.spec, seed=SEED)
baseline = MLPBaseline(e.spec, reg_coef=1e-3, batch_size=64, epochs=2, learn_rate=1e-3)

# agent = NPG(e, policy, baseline, normalized_step_size=0.005, seed=SEED, save_logs=True)

optimizer = fisher_optim.NGD_CG(policy.trainable_params,
                                lr=0.005,
                                decay=False,
                                shrunk=False,
                                lanczos_iters=20,
                                batch_size=500,
                                ascend=True)

# optimizer = fisher_optim.Adaptive_NGD_CG(policy.trainable_params,
#                                 betas=(0.1, 0.1),
#                                 lr=0.005,
#                                 decay=False,
#                                 shrunk=False,
#                                 lanczos_iters=4,
#                                 batch_size=5000,
#                                 ascend=True)

from mjrl.algos.npg_cg_delta import NPG
agent = NPG(e, policy, baseline, optimizer, seed=SEED, save_logs=False)

ts = timer.time()
# 12 = angd lr 0.005, beta (0.1, 0.1)
# 13 = ngd lr 0.005
# 14 = 12 with decay off.
# 15 = 13 with decay off.
train_agent(job_name='results/inverted_pend_lin_test_exp15',
            agent=agent,
            seed=SEED,
            niter=100,
            gamma=0.995,
            gae_lambda=0.97,
            num_cpu=1,
            sample_mode='trajectories',
            num_traj=10,
            save_freq=5,
            evaluation_rollouts=5)
print("time taken for NN policy training = %f" % (timer.time()-ts))


# # Linear policy
# # ==================================
# # e = GymEnv('mjrl_swimmer-v0')
# e = GymEnv('LunarLanderContinuous-v2')
# policy = LinearPolicy(e.spec, seed=SEED)
# baseline = MLPBaseline(e.spec, reg_coef=1e-3, batch_size=64, epochs=2, learn_rate=1e-3)
# # from mjrl.algos.npg_cg import NPG
# # agent = NPG(e, policy, baseline, seed=SEED, save_logs=True)
#
# optimizer = fisher_optim.NGD_CG(policy.trainable_params,
#                                 lr=0.001,
#                                 shrunk=False,
#                                 lanczos_iters=20,
#                                 batch_size=500)
# from mjrl.algos.npg_cg_delta import NPG
# agent = NPG(e, policy, baseline, optimizer, seed=SEED, save_logs=False)
#
# ts = timer.time()
# train_agent(job_name='swimmer_linear_exp1',
#             agent=agent,
#             seed=SEED,
#             niter=50,
#             gamma=0.995,
#             gae_lambda=0.97,
#             num_cpu=1,
#             sample_mode='trajectories',
#             num_traj=10,
#             save_freq=5,
#             evaluation_rollouts=5)
# print("time taken for linear policy training = %f" % (timer.time()-ts))
