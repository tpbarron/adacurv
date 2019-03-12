
import tensorflow as tf

class NGDOptimizer(tf.train.GradientDescentOptimizer):

    def __init__(self,
                learning_rate=0.1,
                name="NGD"):
        super(NGDOptimizer, self).__init__(learning_rate, name=name)

    def minimize(self,
                   loss,
                   global_step=None,
                   var_list=None,
                   gate_gradients=tf.train.Optimizer.GATE_OP,
                   aggregation_method=None,
                   colocate_gradients_with_ops=True,
                   name=None,
                   grad_loss=None,
                   **kwargs):
        # This method has the same general arguments as the minimize methods in
        # standard optimizers do.
        return super(NGDOptimizer, self).minimize(
            loss,
            global_step=global_step,
            var_list=var_list,
            gate_gradients=gate_gradients,
            aggregation_method=aggregation_method,
            colocate_gradients_with_ops=colocate_gradients_with_ops,
            name=name,
            grad_loss=grad_loss,
            **kwargs)

    def _cg_iter(self):
        pass

    def _cg(self, grads_and_vars):

        # Reset deltas
        # While Cond
        # - Perform iter of CG

        with tf.name_scope('conjugate_gradient'):
            cg_update_ops = []

            Ax = self._Hv(gradients, self.cg_delta)

            b = [-grad for grad in gradients]
            bAx = [b - Ax for b, Ax  in zip(b, Ax)]

            condition = tf.equal(self.cg_step, 0)
            r = [tf.cond(condition, lambda: tf.assign(r,  bax),
                lambda: r) for r, bax  in zip(self.residuals, bAx)]

            d = [tf.cond(condition, lambda: tf.assign(d, r),
                lambda: d) for  d, r in zip(self.directions, r)]

            Ad = self.__Hv(gradients, d)
            residual_norm = tf.reduce_sum([tf.reduce_sum(r**2) for r in r])

            alpha = tf.reduce_sum([tf.reduce_sum(d * ad) for d, ad in zip(d, Ad)])
            alpha = residual_norm / alpha

            beta = tf.reduce_sum([tf.reduce_sum((r - alpha * ad)**2) for r, ad in zip(r, Ad)])

            self.beta = beta
            beta = beta / residual_norm

            for i, delta  in reversed(list(enumerate(self.cg_delta))):
                update_delta = tf.assign(delta, delta + alpha * d[i], name='update_delta')
                update_residual = tf.assign(self.residuals[i], r[i] - alpha * Ad[i],
                    name='update_residual')
                p = 1.0
                update_direction = tf.assign(self.directions[i],
                    p * (r[i] - alpha * Ad[i]) + beta * d[i], name='update_direction')
                cg_update_ops.append(update_delta)
                cg_update_ops.append(update_residual)
                cg_update_ops.append(update_direction)

            with tf.control_dependencies(cg_update_ops):
                cg_update_ops.append(tf.assign_add(self.cg_step, 1))
            cg_op = tf.group(*cg_update_ops)

            dl = tf.reduce_sum([tf.reduce_sum(0.5*(delta*ax) + grad*delta)
                for delta, grad, ax in zip(self.cg_delta, gradients, Ax)])

        return cg_op, residual_norm, dl


    def _Hv(self, grads, vec):
        """ Computes Hessian vector product.

        grads: list of Tensorflow tensor objects
            Network gradients.
        vec: list of Tensorflow tensor objects
            Vector that is multiplied by the Hessian.

        return: list of Tensorflow tensor objects
            Result of multiplying Hessian by vec. """
        grad_v = [tf.reduce_sum(g * v) for g, v in zip(grads, vec)]
        Hv = tf.gradients(grad_v, self.W, stop_gradients=vec)
        Hv = [hv + self.damp_pl * v for hv, v in zip(Hv, vec)]
        return Hv

    def compute_gradients(self,
                        loss,
                        var_list=None,
                        gate_gradients=tf.train.Optimizer.GATE_OP,
                        aggregation_method=None,
                        colocate_gradients_with_ops=True,
                        grad_loss=None,
                        **kwargs):
        # This method has the same general arguments as the minimize methods in
        # standard optimizers do.
        grads_and_vars = super(NGDOptimizer, self).compute_gradients(
            loss=loss,
            var_list=var_list,
            gate_gradients=gate_gradients,
            aggregation_method=aggregation_method,
            colocate_gradients_with_ops=colocate_gradients_with_ops,
            grad_loss=grad_loss,
            **kwargs)

        grads_and_vars = self._cg(grads_and_vars)
        return grads_and_vars

    # def apply_gradients(self, grads_and_vars, *args, **kwargs):
    #     grads_and_vars = list(grads_and_vars)
    #
    #     # Update trainable variables with this step, applying self._learning_rate.
    #     apply_op = super(NGDOptimizer, self).apply_gradients(grads_and_vars,
    #                                                       *args,
    #                                                       **kwargs)
    #     with tf.control_dependencies([apply_op]):
    #         # Update the main counter
    #         return tf.group(self._counter.assign(self._counter + 1))
