
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
import pickle
import fisher.optim as fisher_optim
from mjrl.samplers import base_sampler

decay = False

def build_log_dir(tag, variant):
    seed, env, algo, optim, curv_type, lr, batch_size, cg_iters, cg_residual_tol, cg_prev_init_coef, \
        cg_precondition_empirical, cg_precondition_regu_coef, cg_precondition_exp,  \
        shrinkage_method, lanczos_amortization, lanczos_iters, approx_adaptive, betas, use_nn_policy, total_samples = variant
    beta1, beta2 = betas

    dir = os.path.join('results', tag)
    # dir = os.path.join(dir, algo)
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

    dir = os.path.join(dir, "batch_size_"+str(batch_size))
    dir = os.path.join(dir, "lr_"+str(lr))

    if optim in ["natural_adam", "natural_amsgrad"]:
        dir = os.path.join(dir, "betas"+str(beta1)+"_"+str(beta2))

    dir = os.path.join(dir, str(seed))
    return dir

def launch_job(tag, variant):
    seed, env, algo, optim, curv_type, lr, batch_size, cg_iters, cg_residual_tol, cg_prev_init_coef, \
        cg_precondition_empirical, cg_precondition_regu_coef, cg_precondition_exp,  \
        shrinkage_method, lanczos_amortization, lanczos_iters, approx_adaptive, betas, use_nn_policy, total_samples = variant
    beta1, beta2 = betas

    iters = int(total_samples / batch_size)
    save_dir = build_log_dir(tag, variant)

    print ("Save: ", save_dir)
    save_dir = "/Users/trevorbarron/Documents/dev.nosync/thesis/adacurv/experiments/mjrl/results/results_serv/"
    policy_path = os.path.join(save_dir, 'best_policy.pickle')
    # policy_path = os.path.join(save_dir, 'iterations/best_policy.pickle')
    with open(policy_path, 'rb') as f:
        policy = pickle.load(f)
    print (policy)

    e = GymEnv('BasketballEnvRendered-v0')

    N = 10
    T = 250
    paths = base_sampler.do_rollout(N, policy, T, e, None)
    for p in paths:
        print (p['rewards'].sum())

if __name__ == "__main__":
    # tag = 'test'
    # seed x envs x algo x optim x shrunk(bool), lanczos k x batch size x lr x approx adaptive x betas x (use nn policy)
    # variant = [1, 'BasketballEnv-v0', 'trpo', 'ngd', False, 0, 5000, 0.0, False, (0.0, 0.0), True, 1000000]
    # variant = [1, 'BasketballEnv-v0', 'trpo', 'natural_adam', False, 0, 5000, 0.0, False, (0.1, 0.1), True, 250000]

    # variant = [1, 'BasketballEnv-v0', 'trpo', 'ngd', False, 0, 5000, 0.0, False, (0.0, 0.0), True, 250000]
    # variant = [1, 'BasketballEnv-v0', 'trpo', 'natural_adam', True, 5, 5000, 0.0, False, (0.1, 0.1), True, 250000]



    # tag = 'test'
    # variant = [1, 'BasketballEnv-v0', 'trpo', 'ngd', False, 0, 1000, 0.0, False, (0.0, 0.0), True, 250000]

    # tag = 'bball'
    # variant = [1, 'BasketballEnv-v0', 'trpo', 'ngd', False, 0, 5000, 0.0, False, (0.0, 0.0), True, 250000]
    # variant = [1, 'BasketballEnv-v0', 'trpo', 'ngd', False, 0, 1000, 0.0, False, (0.0, 0.0), True, 250000]
    # variant = [1, 'BasketballEnv-v0', 'trpo', 'ngd', False, 0, 5000, 0.0, False, (0.0, 0.0), True, 1000000]

    # tag = 'bball_hoop1.5'
    # variant = [1, 'BasketballEnv-v0', 'trpo', 'ngd', False, 0, 1000, 0.0, False, (0.0, 0.0), True, 500000]

    # tag = 'bball_hoop1.5_posctrl'
    # variant = [1, 'BasketballEnv-v0', 'trpo', 'ngd', False, 0, 5000, 0.0, False, (0.0, 0.0), True, 500000]


    # tag = 'bball_hoop1.5_torqctrl'
    # variant = [1, 'BasketballEnv-v0', 'trpo', 'ngd', False, 0, 1000, 0.0, False, (0.0, 0.0), True, 500000]

    # tag = 'bball_hoop1.5_velctrl_botharms'
    # variant = [1, 'BasketballEnv-v0', 'trpo', 'ngd', False, 0, 1000, 0.0, False, (0.0, 0.0), True, 500000]

    # tag = 'bball_hoop1.5_velctrl_botharms_angle45'
    # variant = [1, 'BasketballEnv-v0', 'trpo', 'ngd', False, 0, 5000, 0.0, False, (0.0, 0.0), True, 1000000]

    # tag = 'bball_hoop1.5_velctrl_botharms_angle55'
    # # variant = [1, 'BasketballEnv-v0', 'trpo', 'ngd', False, 0, 5000, 0.0, False, (0.0, 0.0), True, 1000000]
    # variant = [1, 'BasketballEnv-v0', 'trpo', 'natural_adam', False, 0, 5000, 0.0, False, (0.0, 0.0), True, 1000000]

    # tag = 'bball_hard_velctrl_botharms_angle55'
    # variant = [1, 'BasketballEnvHard-v0', 'trpo', 'ngd', False, 0, 5000, 0.0, False, (0.0, 0.0), True, 1000000]

    tag = 'pybullet_sample_mode_bball'
    variant = [1,                       # seed
               'BasketballEnv-v0',      # env
               'npg',                   # alg
               'ngd',          # optim
               'fisher',                # curv type
               0.01,                     # lr
               5000,                    # batch size
               10,                      # cg iters
               1e-10,                   # cg_res tol
               0.0,                     # cg_prev init coef
               False,                   # cg_prec emp
               0.0,                     # cg_precondition_regu_coef
               0.0,                     # cg_prec exp
               None,                    # shrinkage_method
               0,                       # lanzcos_amortization
               0,                       # lanzcos_iters
               False,                   # approx adaptive
               (0.1, 0.1),              # betas
               True,                    # use nn
               1000000]                 # total_samples

    launch_job(tag, variant)
