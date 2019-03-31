
import tensorflow as tf

from adacurv.tf.utils import cg
from adacurv.tf.utils import curvature_matrix_vector_products as cmvps

class NaturalAdamNGDOptimizer:

    def __init__(self,
                learning_rate=0.001,
                beta1=0.1,
                beta2=0.1,
                cg_decay=0.0,
                name="NaturalAdamNGD"):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.cg_decay = cg_decay
        self.vars = tf.trainable_variables()

        self.lagged_vars = [tf.Variable(v.initialized_value() + tf.random.normal(shape=v.shape) * 0.001) for v in self.vars]
        self.tmp_vars = [tf.Variable(v.initialized_value()) for v in self.vars]

        self.cg_step, \
            self.cg_delta, \
            self.directions, \
            self.residuals, \
            self.residual_norm = cg.initialize_cg_vars(self.vars)

        self.reset_cg_delta = cg.create_reset_cg_delta_op(self.cg_delta, self.cg_decay)
        self.reset_cg_step = cg.create_reset_cg_step_op(self.cg_step)

        self.Cvp_fn = cmvps.Fvp
        # self.Cvp_fn = cmvps.GNvp

        self.m = [tf.Variable(tf.constant(0, shape=v.shape, dtype=tf.float32)) for v in self.vars]
        self.m_hat = [tf.Variable(tf.constant(0, shape=v.shape, dtype=tf.float32)) for v in self.vars]
        self.cg_delta_prev = [tf.Variable(tf.constant(0, shape=v.shape, dtype=tf.float32)) for v in self.vars]

        self.adap_n = tf.Variable(tf.constant(10, dtype=tf.int32))
        self.adap_step = tf.Variable(tf.constant(0, dtype=tf.int32))
        self.objectives = tf.Variable(tf.constant(0, shape=(10,), dtype=tf.float32))

    def assign_vars_to_tmp_op(self):
        assign_to_tmp_ops = [tf.assign(tv, v) for tv, v in zip(self.tmp_vars, self.vars)]
        return assign_to_tmp_ops

    def assign_tmp_to_vars_op(self):
        assign_to_vars_ops = [tf.assign(v, tv) for v, tv in zip(self.vars, self.tmp_vars)]
        return assign_to_vars_ops

    def assign_lagged_vars_to_vars(self):
        assign_to_vars_ops = [tf.assign(v, lv) for v, lv in zip(self.vars, self.lagged_vars)]
        return assign_to_vars_ops

    def assign_interpolated_to_vars(self, alpha):
        assign_interpolated_vars_ops = [tf.assign(v, alpha * lv + (1-alpha) * v) for v, lv in zip(self.vars, self.lagged_vars)]
        return assign_interpolated_vars_ops

    def assign_interpolated_to_lagged_vars(self, alpha):
        assign_interpolated_vars_ops = [tf.assign(lv, alpha * lv + (1-alpha) * v) for v, lv in zip(self.vars, self.lagged_vars)]
        return assign_interpolated_vars_ops

    def assign_delta_to_prev(self):
        assign_to_cg_delta_prev_ops = [tf.assign(cg_prev, cg) for cg_prev, cg in zip(self.cg_delta_prev, self.cg_delta)]
        return assign_to_cg_delta_prev_ops

    def assign_grads_to_prev(self, grads):
        assign_grad_to_cg_delta_prev_ops = [tf.assign(cg_prev, g) for cg_prev, g in zip(self.cg_delta_prev, grads)]
        return assign_grad_to_cg_delta_prev_ops


    def adaptive_grad_update_op(self, grads):
        gradient_moving_average_op = [tf.assign(mi, (self.beta1 * mi + (1-self.beta1) * gi)) for mi, gi in zip(self.m, grads)]
        return gradient_moving_average_op

    def adaptive_grad_bias_correct(self, g_step):
        gradient_bias_correct = [tf.assign(mi_hat, mi / (1.0-self.beta1**tf.cast(g_step+1, tf.float32))) for mi_hat, mi in zip(self.m_hat, self.m)]
        return gradient_bias_correct

    def approx_adaptive_update_op(self, g_step):
        shift_by_beta_ops = [tf.assign(lv, self.beta2 * lv + (1-self.beta2) * v) for lv, v in zip(self.lagged_vars, self.vars)]
        approx_adap_op = tf.cond(g_step > 0, lambda: shift_by_beta_ops, lambda: self.lagged_vars)
        return approx_adap_op

    def _weighted_fvp_both(self, loss, z):
        fvp_current = self.Cvp_fn(loss, z, self.vars, self.cg_delta_prev)
        with tf.control_dependencies(self.assign_vars_to_tmp_op()):
            with tf.control_dependencies(self.assign_lagged_vars_to_vars()):
                fvp_lagged = self.Cvp_fn(loss, z, self.vars, self.cg_delta_prev)

        with tf.control_dependencies(self.assign_tmp_to_vars_op()):
            fvp_lagged_beta2 = [self.beta2 * fvp_lag for fvp_lag in fvp_lagged]
            fvp_current_beta2 = [(1-self.beta2) * fvp_cur for fvp_cur in fvp_current]
            obj = [fvp1 + fvp2 for fvp1, fvp2 in zip(fvp_lagged_beta2, fvp_current_beta2)]
        return obj

    def _weighted_fvp_one(self, loss, z):
        fvp_current = self.Cvp_fn(loss, z, self.vars, self.cg_delta_prev)
        fvp_current_beta2 = [(1-self.beta2) * fvp_cur for fvp_cur in fvp_current]
        return fvp_current_beta2

    def fvp_objective(self, g_step, loss, z):
        obj = tf.cond(g_step > 0, lambda: self._weighted_fvp_both(loss, z), lambda: self._weighted_fvp_one(loss, z))
        return obj

    def optim_adaptive_update_op(self, g_step, loss, z, n=10, eps=1e-5):
        # for each intermediate test.
        with tf.control_dependencies([tf.assign(self.adap_step, 0)]):
            obj = self.fvp_objective(g_step, loss, z)
            # adap_n = tf.Variable(tf.constant(n, dtype=tf.int32))
            # adap_step = tf.Variable(tf.constant(0, dtype=tf.int32))
            # objectives = tf.Variable(tf.constant(0, shape=(10,), dtype=tf.float32))
            alphas = tf.linspace(0.0+eps, 1.0-eps, self.adap_n, name='linspace')

        # def _eval_variant(alpha):
        #     with tf.control_dependencies(self.assign_vars_to_tmp_op()):
        #         with tf.control_dependencies(self.assign_interpolated_to_vars(alpha)): #[tf.print(alpha)] + [tf.print(self.vars), tf.print(self.lagged_vars)] + [tf.print(self.assign_interpolated_to_vars(alpha))]):
        #             # with tf.control_dependencies([tf.print(self.vars)]):
        #             out = self.Cvp_fn(loss, z, self.vars, self.cg_delta_prev)
        #             with tf.control_dependencies(self.assign_tmp_to_vars_op()): # + [tf.print(tf.sqrt(tf.reduce_sum([tf.reduce_sum((fvp_a - fvp_b)**2.0) for fvp_a, fvp_b in zip(out, obj)])))]): # + [tf.print(alpha)]):
        #                 j = tf.sqrt(tf.reduce_sum([tf.reduce_sum((fvp_a - fvp_b)**2.0) for fvp_a, fvp_b in zip(out, obj)]))
        #                 return j
        #
        # # TODO: find way to parallelize this. perhaps copy model?
        # objectives = tf.map_fn(_eval_variant,
        #                        alphas,
        #                        dtype=tf.float32,
        #                        parallel_iterations=1)
        # alpha_min = alphas[tf.argmin(objectives)]
        # with tf.control_dependencies([tf.print(alpha_min)] + [tf.print(objectives)]):
        #     return self.assign_interpolated_to_lagged_vars(alpha_min)

        def adap_cond(adap_step, alphas): #, objectives):
            with tf.control_dependencies([tf.print(adap_step)]):
                return tf.less(adap_step, self.adap_n)

        def adap_body(adap_step, alphas): #, objectives):
            alpha = alphas[adap_step]
            with tf.control_dependencies(self.assign_vars_to_tmp_op()):
                with tf.control_dependencies(self.assign_interpolated_to_vars(alpha)):
                    out = self.Cvp_fn(loss, z, self.vars, self.cg_delta_prev)
                    with tf.control_dependencies(self.assign_tmp_to_vars_op()):
                        j = tf.sqrt(tf.reduce_sum([tf.reduce_sum((fvp_a - fvp_b)**2.0) for fvp_a, fvp_b in zip(out, obj)]))
                        with tf.control_dependencies([tf.assign(self.objectives[adap_step], j)]):
                            with tf.control_dependencies([tf.print(alpha), tf.print(adap_step)]):
                                return adap_step + 1, alphas

        adap_op = tf.while_loop(
            adap_cond,
            adap_body,
            (self.adap_step, alphas,), #, objectives),
            parallel_iterations=1)

        with tf.control_dependencies([adap_op[0]]):
            final_objectives = self.objectives #adap_op[2]
            alpha_min = alphas[tf.argmin(final_objectives)]
            with tf.control_dependencies([tf.print(alpha_min)] + [tf.print(self.objectives)]):
                return self.assign_interpolated_to_lagged_vars(alpha_min)
            # return self.assign_interpolated_to_lagged_vars(alpha_min)

    def normalized_step_size(self):
        return tf.sqrt(tf.abs(self.learning_rate / (tf.reduce_sum([tf.reduce_sum(m_h * cg_d) for m_h, cg_d in zip(self.m_hat, self.cg_delta)]) + 1e-20) ))

    def _print(self, tag, tensor):
        return [tf.print(tf.constant(tag)), tf.print(tensor)]
        # return [tf.print(tensor)]

    def minimize(self,
                   loss,
                   z,
                   global_step=None,
                   var_list=None,
                   gate_gradients=tf.train.Optimizer.GATE_OP,
                   aggregation_method=None,
                   colocate_gradients_with_ops=True,
                   name=None,
                   grad_loss=None,
                   **kwargs):

        approx_adap_ops = self.approx_adaptive_update_op(global_step)
        # optim_adap_ops = self.optim_adaptive_update_op(global_step, loss, z)
        set_vars_to_tmp = self.assign_vars_to_tmp_op()
        set_lagged_to_vars = self.assign_lagged_vars_to_vars()

        grads = tf.gradients(loss, self.vars,
            colocate_gradients_with_ops=colocate_gradients_with_ops,
            gate_gradients=gate_gradients,
            aggregation_method=aggregation_method)

        # print grads
        # with tf.control_dependencies(self._print("grads", grads)):
        adap_grad_ops = self.adaptive_grad_update_op(grads)
        # print m
        with tf.control_dependencies(adap_grad_ops):
            adap_bias_corr_ops = self.adaptive_grad_bias_correct(global_step)
        # print m_hat

        with tf.control_dependencies(adap_bias_corr_ops):
            maybe_assign_grads_to_cg_prev = tf.cond(tf.equal(global_step, 0), lambda: self.assign_grads_to_prev(self.m_hat), lambda: self.m_hat)

        with tf.control_dependencies([self.reset_cg_step, self.reset_cg_delta] + maybe_assign_grads_to_cg_prev):
            # with tf.control_dependencies(self.optim_adaptive_update_op(global_step, loss, z)):
            with tf.control_dependencies(approx_adap_ops):
                with tf.control_dependencies(set_vars_to_tmp):
                    with tf.control_dependencies(set_lagged_to_vars):
                        bias2_correction = 1.0-self.beta2**tf.cast(global_step+1, tf.float32)
                        cg_inputs = [self.Cvp_fn, adap_bias_corr_ops, loss, z, self.vars, self.cg_step, self.cg_delta, self.directions, self.residuals, self.residual_norm]
                        cg_op = cg.cg_solve(*cg_inputs, bias2_correction=bias2_correction)

        with tf.control_dependencies(cg_op[1] + cg_op[2] + cg_op[3] + [cg_op[4]]):
            # with tf.control_dependencies(self.assign_delta_to_prev()):# + [tf.print(self.cg_delta)]):
            with tf.control_dependencies(self.assign_tmp_to_vars_op() + self.assign_delta_to_prev()):
                var_update_ops = []
                norm_step = self.normalized_step_size()
                for i in range(len(self.vars)):
                    var_update = tf.assign(self.vars[i], self.vars[i] - norm_step * self.cg_delta[i])
                    var_update_ops.append(var_update)

                g_step_op = tf.assign_add(global_step, 1)
                train_op = tf.group(var_update_ops + [g_step_op])

        return train_op
