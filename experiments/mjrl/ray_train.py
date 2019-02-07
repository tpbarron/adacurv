import os
import time as timer

from mjrl.utils.gym_env import GymEnv
from mjrl.policies.gaussian_mlp import MLP
from mjrl.policies.gaussian_linear import LinearPolicy
from mjrl.baselines.quadratic_baseline import QuadraticBaseline
from mjrl.baselines.mlp_baseline import MLPBaseline
from mjrl.utils.train_agent import train_agent
from mjrl.algos.npg_cg import NPG
from mjrl.algos.trpo import TRPO
import mjrl.envs

import fisher.optim as fisher_optim

decay = False

def build_log_dir(tag, variant):
    seed, env, algo, optim, shrunk, lanczos_iters, batch_size, lr, approx_adaptive, betas, use_nn_policy, total_samples = variant
    beta1, beta2 = betas

    dir = os.path.join('results', tag)
    dir = os.path.join(dir, env)
    dir = os.path.join(dir, algo)
    dir = os.path.join(dir, optim)
    if approx_adaptive:
        dir = os.path.join(dir, "approx_adaptive")
    else:
        dir = os.path.join(dir, "optim_adaptive")

    if shrunk:
        dir = os.path.join(dir, "shrunk_true")
        dir = os.path.join(dir, "lanczos_iters_"+str(lanczos_iters))
    else:
        dir = os.path.join(dir, "shrunk_false")

    if use_nn_policy:
        dir = os.path.join(dir, "nn_policy")
    else:
        dir = os.path.join(dir, "lin_policy")

    dir = os.path.join(dir, "total_samples_"+str(total_samples))
    dir = os.path.join(dir, "batch_size_"+str(batch_size))
    dir = os.path.join(dir, "lr_"+str(lr))

    if optim in ["natural_adam", "natural_amsgrad"]:
        dir = os.path.join(dir, "betas"+str(beta1)+"_"+str(beta2))

    dir = os.path.join(dir, str(seed))
    return dir

def launch_job(tag, variant):
    seed, env, algo, optim, shrunk, lanczos_iters, batch_size, lr, approx_adaptive, betas, use_nn_policy, total_samples = variant
    iters = int(total_samples / batch_size)

    # NN policy
    # ==================================
    e = GymEnv(env)
    if use_nn_policy:
        policy = MLP(e.spec, hidden_sizes=(64,), seed=seed)
    else:
        policy = LinearPolicy(e.spec, seed=seed)
    baseline = MLPBaseline(e.spec, reg_coef=1e-3, batch_size=64, epochs=2, learn_rate=1e-3)
    # agent = NPG(e, policy, baseline, normalized_step_size=0.005, seed=SEED, save_logs=True)

    if optim == 'ngd':
        optimizer = fisher_optim.NGD(policy.trainable_params,
                                     lr=lr,
                                     decay=decay,
                                     shrunk=shrunk,
                                     lanczos_iters=lanczos_iters,
                                     batch_size=batch_size,
                                     ascend=True)
    elif optim == 'natural_adam':
        optimizer = fisher_optim.NaturalAdam(policy.trainable_params,
                                             betas=betas,
                                             lr=lr,
                                             decay=decay,
                                             shrunk=shrunk,
                                             lanczos_iters=lanczos_iters,
                                             batch_size=batch_size,
                                             ascend=True,
                                             assume_locally_linear=approx_adaptive)
    elif optim == 'natural_adagrad':
        optimizer = fisher_optim.NaturalAdagrad(policy.trainable_params,
                                                lr=lr,
                                                decay=decay,
                                                shrunk=shrunk,
                                                lanczos_iters=lanczos_iters,
                                                batch_size=batch_size,
                                                ascend=True,
                                                assume_locally_linear=approx_adaptive)
    elif optim == 'natural_amsgrad':
        optimizer = fisher_optim.NaturalAmsgrad(policy.trainable_params,
                                                betas=betas,
                                                lr=lr,
                                                decay=decay,
                                                shrunk=shrunk,
                                                lanczos_iters=lanczos_iters,
                                                batch_size=batch_size,
                                                ascend=True,
                                                assume_locally_linear=approx_adaptive)

    if algo == 'trpo':
        from mjrl.algos.trpo_delta import TRPO
        agent = TRPO(e, policy, baseline, optimizer, seed=seed, save_logs=True)
        # agent = TRPO(e, policy, baseline, seed=seed, save_logs=True)
    else:
        from mjrl.algos.npg_cg_delta import NPG
        agent = NPG(e, policy, baseline, optimizer, seed=seed, save_logs=True)

    save_dir = build_log_dir(tag, variant)
    try:
        os.makedirs(save_dir)
    except:
        pass

    # print ("Iters:", iters, ", num_traj: ", str(batch_size//1000))
    train_agent(job_name=save_dir,
                agent=agent,
                seed=seed,
                niter=iters,
                gamma=0.995,
                gae_lambda=0.97,
                num_cpu=1,
                sample_mode='samples',
                num_samples=batch_size,
                save_freq=5,
                evaluation_rollouts=5,
                verbose=True)

if __name__ == "__main__":
    # Test
    # tag = 'walker2d_test'
    tag = 'bball'
    # seed x envs x algo x optim x shrunk(bool), lanczos k x batch size x lr x approx adaptive x betas x (use nn policy)
    # variant = [1, 'InvertedPendulumBulletEnv-v0', 'trpo', 'natural_adam', False, 0, 5000, 0.01, False, (0.0, 0.0), False, 200000]
    # variant = [1, 'HalfCheetahBulletEnv-v0', 'natural_adam', False, 100, 5000, 0.05, False, (0.1, 0.1), True, 250000]
    # variant = [1, 'MinitaurBulletEnv-v0', 'trpo', 'natural_amsgrad', False, 0, 5000, 0.01, False, (0.0, 0.0), False, 100000]

    # 41.03s
    # variant = [1, 'Walker2DBulletEnv-v0', 'trpo', 'natural_adam', True, 5, 5000, 0.001, False, (0.1, 0.1), True, 25000]
    # 25.08s
    # variant = [1, 'Walker2DBulletEnv-v0', 'trpo', 'natural_adam', False, 5, 5000, 0.001, False, (0.1, 0.1), True, 25000]
    # 25.08s
    # variant = [1, 'Walker2DBulletEnv-v0', 'trpo', 'natural_adam', False, 5, 5000, 0.001, True, (0.1, 0.1), True, 25000]
    # 20.14s
    # variant = [1, 'Walker2DBulletEnv-v0', 'trpo', 'ngd', False, 5, 5000, 0.001, False, (0.1, 0.1), True, 25000]
    # 35.17s
    # variant = [1, 'Walker2DBulletEnv-v0', 'trpo', 'ngd', True, 5, 5000, 0.001, False, (0.1, 0.1), True, 25000]


    # variant = [1, 'BasketballEnv-v0', 'trpo', 'natural_adam', True, 5, 5000, 0.0, False, (0.1, 0.1), True, 250000]
    # variant = [1, 'BasketballEnv-v0', 'trpo', 'ngd', False, 0, 5000, 0.0, False, (0.0, 0.0), True, 250000]
    variant = [1, 'BasketballEnv-v0', 'trpo', 'ngd', False, 0, 1000, 0.0, False, (0.0, 0.0), True, 250000]
    launch_job(tag, variant)
