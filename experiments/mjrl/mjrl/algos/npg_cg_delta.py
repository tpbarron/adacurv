import copy
import time as timer
import logging
logging.disable(logging.CRITICAL)

import numpy as np
import scipy as sp
import scipy.sparse.linalg as spLA

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils import vector_to_parameters, parameters_to_vector

# samplers
import mjrl.samplers.trajectory_sampler as trajectory_sampler
import mjrl.samplers.batch_sampler as batch_sampler

# utility functions
import mjrl.utils.process_samples as process_samples
from mjrl.utils.logger import DataLog
from mjrl.utils.cg_solve import cg_solve
from mjrl.algos.batch_reinforce import BatchREINFORCE
from adacurv.torch.utils.convert_gradients import gradients_to_vector, vector_to_gradients


class NPG(BatchREINFORCE):
    def __init__(self, env, policy, baseline, optim,
                 normalized_step_size=0.01,
                 const_learn_rate=None,
                 FIM_invert_args={'iters': 10, 'damping': 1e-4},
                 hvp_sample_frac=1.0,
                 seed=None,
                 save_logs=False,
                 kl_dist=None):
        """
        All inputs are expected in mjrl's format unless specified
        :param normalized_step_size: Normalized step size (under the KL metric). Twice the desired KL distance
        :param kl_dist: desired KL distance between steps. Overrides normalized_step_size.
        :param const_learn_rate: A constant learn rate under the L2 metric (won't work very well)
        :param FIM_invert_args: {'iters': # cg iters, 'damping': regularization amount when solving with CG
        :param hvp_sample_frac: fraction of samples (>0 and <=1) to use for the Fisher metric (start with 1 and reduce if code too slow)
        :param seed: random seed
        """

        self.env = env
        self.policy = policy
        self.baseline = baseline
        self.optim = optim


        self.alpha = const_learn_rate
        self.n_step_size = normalized_step_size if kl_dist is None else 2.0 * kl_dist
        self.seed = seed
        self.save_logs = save_logs
        self.FIM_invert_args = FIM_invert_args
        self.hvp_subsample = hvp_sample_frac
        self.running_score = None
        self.n_steps = 0
        if save_logs: self.logger = DataLog()


    def policy_kl_fn(self, policy, obs, act):
        old_dist_info = policy.old_dist_info(obs, act)
        new_dist_info = policy.new_dist_info(obs, act)
        mean_kl = policy.mean_kl(new_dist_info, old_dist_info)
        return mean_kl

    def kl_closure(self, policy, observations, actions, kl_fn):
        def func(params):
            old_params = policy.get_param_values()
            params = parameters_to_vector(params).data.numpy()
            policy.set_param_values(params, set_new=True, set_old=True)
            f = kl_fn(policy, observations, actions)

            tmp_params = policy.trainable_params
            policy.set_param_values(old_params, set_new=True, set_old=True)
            return f, tmp_params
        return func

    def HVP(self, policy, observations, actions, vec, regu_coef=None):
        regu_coef = self.FIM_invert_args['damping'] if regu_coef is None else regu_coef
        # vec = Variable(torch.from_numpy(vector).float(), requires_grad=False)
        if self.hvp_subsample is not None and self.hvp_subsample < 0.99:
            num_samples = observations.shape[0]
            rand_idx = np.random.choice(num_samples, size=int(self.hvp_subsample*num_samples))
            obs = observations[rand_idx]
            act = actions[rand_idx]
        else:
            obs = observations
            act = actions
        old_dist_info = policy.old_dist_info(obs, act)
        new_dist_info = policy.new_dist_info(obs, act)
        mean_kl = policy.mean_kl(new_dist_info, old_dist_info)
        grad_fo = torch.autograd.grad(mean_kl, policy.trainable_params, create_graph=True)
        flat_grad = torch.cat([g.contiguous().view(-1) for g in grad_fo])
        h = torch.sum(flat_grad*vec)
        hvp = torch.autograd.grad(h, policy.trainable_params)
        hvp_flat = torch.cat([g.contiguous().view(-1).data for g in hvp])
        # hvp_flat = np.concatenate([g.contiguous().view(-1).data.numpy() for g in hvp])

        hvp_res = hvp_flat + regu_coef*vec
        return hvp_res

    def build_Hvp_eval(self, policy, inputs, regu_coef=None):
        def eval(theta, v):
            policy_tmp = copy.deepcopy(policy)
            policy_tmp.set_param_values(theta.data.numpy())
            full_inp = [policy_tmp] + inputs + [v] + [regu_coef]
            Hvp = self.HVP(*full_inp)
            return Hvp
        return eval

    # ----------------------------------------------------------
    def train_from_paths(self, paths):
        # Concatenate from all the trajectories
        observations = np.concatenate([path["observations"] for path in paths])
        actions = np.concatenate([path["actions"] for path in paths])
        advantages = np.concatenate([path["advantages"] for path in paths])
        # Advantage whitening
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-6)
        # NOTE : advantage should be zero mean in expectation
        # normalized step size invariant to advantage scaling,
        # but scaling can help with least squares

        self.n_steps += len(advantages)

        # cache return distributions for the paths
        path_returns = [sum(p["rewards"]) for p in paths]
        mean_return = np.mean(path_returns)
        std_return = np.std(path_returns)
        min_return = np.amin(path_returns)
        max_return = np.amax(path_returns)
        base_stats = [mean_return, std_return, min_return, max_return]
        self.running_score = mean_return if self.running_score is None else \
                             0.9*self.running_score + 0.1*mean_return  # approx avg of last 10 iters
        if self.save_logs: self.log_rollout_statistics(paths)

        # Keep track of times for various computations
        t_gLL = 0.0
        t_FIM = 0.0

        self.optim.zero_grad()

        # Optimization. Negate gradient since the optimizer is minimizing.
        vpg_grad = -self.flat_vpg(observations, actions, advantages)
        vector_to_gradients(Variable(torch.from_numpy(vpg_grad).float()), self.policy.trainable_params)

        closure = self.kl_closure(self.policy, observations, actions, self.policy_kl_fn)
        info = self.optim.step(closure)
        self.policy.set_param_values(self.policy.get_param_values())

        # Log information
        if self.save_logs:
            self.logger.log_kv('alpha', info['alpha'])
            self.logger.log_kv('delta', info['delta'])
            # self.logger.log_kv('time_vpg', t_gLL)
            # self.logger.log_kv('time_npg', t_FIM)
            # self.logger.log_kv('kl_dist', kl_dist)
            # self.logger.log_kv('surr_improvement', surr_after - surr_before)
            self.logger.log_kv('running_score', self.running_score)
            self.logger.log_kv('steps', self.n_steps)

            try:
                success_rate = self.env.env.env.evaluate_success(paths)
                self.logger.log_kv('success_rate', success_rate)
            except:
                pass

        return base_stats
