
import tensorflow as tf

from adacurv.tf.utils import cg
from adacurv.tf.utils import curvature_matrix_vector_products as cmvps

class NGDOptimizer: #tf.train.GradientDescentOptimizer):

    def __init__(self,
                learning_rate=0.001,
                cg_decay=0.0,
                name="NGD"):
        # super(NGDOptimizer, self).__init__(learning_rate, name=name)
        self.learning_rate = learning_rate
        self.cg_decay = cg_decay
        self.vars = tf.trainable_variables()

        self.cg_step, \
            self.cg_delta, \
            self.directions, \
            self.residuals, \
            self.residual_norm = cg.initialize_cg_vars(self.vars)

        self.reset_cg_delta = cg.create_reset_cg_delta_op(self.cg_delta, self.cg_decay)
        self.reset_cg_step = cg.create_reset_cg_step_op(self.cg_step)

        # self.Cvp_fn = cmvps.GNvp
        self.Cvp_fn = cmvps.Fvp

    def normalized_step_size(self, grads):
        return tf.sqrt(tf.abs(self.learning_rate / (tf.reduce_sum([tf.reduce_sum(m_h * cg_d) for m_h, cg_d in zip(grads, self.cg_delta)]) + 1e-20) ))

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

        grads = tf.gradients(loss, self.vars,
            colocate_gradients_with_ops=colocate_gradients_with_ops,
            gate_gradients=gate_gradients,
            aggregation_method=aggregation_method)

        with tf.control_dependencies([self.reset_cg_step, self.reset_cg_delta]):
            cg_inputs = [self.Cvp_fn, grads, loss, z, self.vars, self.cg_step, self.cg_delta, self.directions, self.residuals, self.residual_norm]
            cg_op = cg.cg_solve(*cg_inputs)

        with tf.control_dependencies(cg_op[1]):
            var_update_ops = []
            for i in range(len(self.vars)):
                var_update = tf.assign_add(self.vars[i], -self.normalized_step_size(grads) * self.cg_delta[i])
                var_update_ops.append(var_update)

            g_step_op = tf.assign_add(global_step, 1)
            train_op = tf.group(var_update_ops + [g_step_op])

        return train_op
