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
    seed, env, algo, optim, curv_type, lr, batch_size, cg_iters, cg_residual_tol, cg_prev_init_coef, \
        cg_precondition_empirical, cg_precondition_regu_coef, cg_precondition_exp,  \
        shrinkage_method, lanczos_amortization, lanczos_iters, approx_adaptive, betas, use_nn_policy, gn_vfn_opt, total_samples = variant
    beta1, beta2 = betas

    dir = os.path.join('results', tag)
    dir = os.path.join(dir, env)
    dir = os.path.join(dir, algo)
    dir = os.path.join(dir, optim)
    if approx_adaptive:
        dir = os.path.join(dir, "approx_adaptive")
    else:
        dir = os.path.join(dir, "optim_adaptive")

    dir = os.path.join(dir, "curv_type_" + curv_type)
    dir = os.path.join(dir, "cg_iters_" + str(cg_iters))
    dir = os.path.join(dir, "cg_residual_tol_" + str(cg_residual_tol))
    dir = os.path.join(dir, "cg_prev_init_coef_" + str(cg_prev_init_coef))

    if cg_precondition_empirical:
        dir = os.path.join(dir, 'cg_precondition_empirical_true')
        dir = os.path.join(dir, "cg_precondition_regu_coef_" + str(cg_precondition_regu_coef))
        dir = os.path.join(dir, "cg_precondition_exp_" + str(cg_precondition_exp))
    else:
        dir = os.path.join(dir, 'cg_precondition_empirical_false')

    if shrinkage_method is not None:
        if shrinkage_method == 'lanzcos':
            dir = os.path.join(dir, "shrunk_true/lanzcos")
            dir = os.path.join(dir, "lanczos_amortization_"+str(lanczos_amortization))
            dir = os.path.join(dir, "lanczos_iters_"+str(lanczos_iters))
        elif shrinkage_method == 'cg':
            dir = os.path.join(dir, "shrunk_true/cg")
    else:
        dir = os.path.join(dir, "shrunk_false")

    if use_nn_policy:
        dir = os.path.join(dir, "nn_policy")
    else:
        dir = os.path.join(dir, "lin_policy")

    if gn_vfn_opt:
        dir = os.path.join(dir, "gn_vfn_opt")
    else:
        dir = os.path.join(dir, "adam_vfn_opt")

    dir = os.path.join(dir, "total_samples_"+str(total_samples))
    dir = os.path.join(dir, "batch_size_"+str(batch_size))
    dir = os.path.join(dir, "lr_"+str(lr))

    if optim in ["natural_adam", "natural_amsgrad"]:
        dir = os.path.join(dir, "betas"+str(beta1)+"_"+str(beta2))

    dir = os.path.join(dir, str(seed))
    return dir

def launch_job(tag, variant):

    seed, env, algo, optim, curv_type, lr, batch_size, cg_iters, cg_residual_tol, cg_prev_init_coef, \
        cg_precondition_empirical, cg_precondition_regu_coef, cg_precondition_exp,  \
        shrinkage_method, lanczos_amortization, lanczos_iters, approx_adaptive, betas, use_nn_policy, gn_vfn_opt, total_samples = variant
    beta1, beta2 = betas

    iters = int(total_samples / batch_size)

    # NN policy
    # ==================================
    e = GymEnv(env)
    if use_nn_policy:
        policy = MLP(e.spec, hidden_sizes=(64,), seed=seed)
    else:
        policy = LinearPolicy(e.spec, seed=seed)
    vfn_batch_size = 256 if gn_vfn_opt else 64
    vfn_epochs = 2 if gn_vfn_opt else 2
    # baseline = MLPBaseline(e.spec, reg_coef=1e-3, batch_size=64, epochs=2, learn_rate=1e-3)
    baseline = MLPBaseline(e.spec, reg_coef=1e-3, batch_size=vfn_batch_size, epochs=2, learn_rate=1e-3, use_gauss_newton=gn_vfn_opt)
    # agent = NPG(e, policy, baseline, normalized_step_size=0.005, seed=SEED, save_logs=True)

    common_kwargs = dict(lr=lr,
                         curv_type=curv_type,
                         cg_iters=cg_iters,
                         cg_residual_tol=cg_residual_tol,
                         cg_prev_init_coef=cg_prev_init_coef,
                         cg_precondition_empirical=cg_precondition_empirical,
                         cg_precondition_regu_coef=cg_precondition_regu_coef,
                         cg_precondition_exp=cg_precondition_exp,
                         shrinkage_method=shrinkage_method,
                         lanczos_amortization=lanczos_amortization,
                         lanczos_iters=lanczos_iters,
                         batch_size=batch_size)

    if optim == 'ngd':
        optimizer = fisher_optim.NGD(policy.trainable_params, **common_kwargs)
    elif optim == 'natural_adam':
        optimizer = fisher_optim.NaturalAdam(policy.trainable_params,
                                             **common_kwargs,
                                             betas=(beta1, beta2),
                                             assume_locally_linear=approx_adaptive)
    elif optim == 'natural_adagrad':
        optimizer = fisher_optim.NaturalAdagrad(policy.trainable_params,
                                                **common_kwargs,
                                                betas=(beta1, beta2),
                                                assume_locally_linear=approx_adaptive)
    elif optim == 'natural_amsgrad':
        optimizer = fisher_optim.NaturalAmsgrad(policy.trainable_params,
                                                **common_kwargs,
                                                betas=(beta1, beta2),
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
                verbose=False) #True)

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
    # variant = [1, 'BasketballEnv-v0', 'trpo', 'ngd', False, 0, 5000, 0.0, False, (0.0, 0.0), True, 1000000]

    # tag = 'bball_hoop1.5_torqctrl'
    # variant = [1, 'BasketballEnv-v0', 'trpo', 'ngd', False, 0, 1000, 0.0, False, (0.0, 0.0), True, 500000]

    tag = 'test' #bball_randhoop1.5_velctrl_botharms_angle55'
    # variant = [1, 'BasketballEnv-v0', 'trpo', 'ngd', False, 0, 5000, 0.0, False, (0.0, 0.0), True, 1000000]
    # variant = [1, 'BasketballEnv-v0', 'trpo', 'natural_adam', False, 0, 5000, 0.0, False, (0.1, 0.1), True, 1000000]

    # # seed, envs, alg, optim, curv_type, lr, batch size, cg_iters, cg_residual_tol, cg_prev_init_coef, cg_precondition_empirical, cg_precondition_regu_coef, cg_precondition_exp
    # shrinkage_methodm, lanzcos_amortization, lanzcos_iters,  approx adaptive, betas, use nn, use gn_vfn_opt, total_samples
    variant = [1, 'BasketballEnvRandomHoop-v0', 'trpo', 'natural_adam', 'fisher', 0.0, 1000, 10, 1e-10, 0.5, True, 0.001, 0.75, None, 0, 0, False, (0.1, 0.1), True, False, 1000000]

    # tag = 'test'
    # # 18.96 ngd- no shrink
    # variant = [1, 'Walker2DBulletEnv-v0', 'trpo', 'ngd', 'fisher', 0.0, 5000, 10, 1e-10, 0.0, False, 0.0, 0.0, None, 0, 0, False, (0.1, 0.1), True, 25000]
    #
    # # 18.44 ngd - shrink 'cg'
    # variant = [1, 'Walker2DBulletEnv-v0', 'trpo', 'ngd', 'fisher', 0.0, 5000, 10, 1e-10, 0.0, False, 0.0, 0.0, 'cg', 0, 0, False, (0.1, 0.1), True, 25000]
    #
    # # 19.40 ngd - shrink 'lanzcos'
    # variant = [1, 'Walker2DBulletEnv-v0', 'trpo', 'ngd', 'fisher', 0.0, 5000, 10, 1e-10, 0.0, False, 0.0, 0.0, 'lanczos', 10, 1, False, (0.1, 0.1), True, 25000]
    #
    # # 21.94 nat adam - shrink 'cg'
    # variant = [1, 'Walker2DBulletEnv-v0', 'trpo', 'natural_adam', 'fisher', 0.0, 5000, 10, 1e-10, 0.0, False, 0.0, 0.0, 'cg', 0, 0, False, (0.1, 0.1), True, 25000]
    #
    # # 21.54 nat adam - no shrink
    # variant = [1, 'Walker2DBulletEnv-v0', 'trpo', 'natural_adam', 'fisher', 0.0, 5000, 10, 1e-10, 0.0, False, 0.0, 0.0, None, 0, 0, False, (0.1, 0.1), True, 25000]
    #
    # # variant = [1, 'Walker2DBulletEnv-v0', 'trpo', 'natural_adam', 'fisher', 0.0, 5000, 10, 1e-10, 0.0, False, 0.0, 0.0, None, 0, 0, False, (0.1, 0.1), True, 1000000]

    launch_job(tag, variant)
