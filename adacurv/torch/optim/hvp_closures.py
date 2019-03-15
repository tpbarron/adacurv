import torch
import torch.nn.functional as F
from torch.autograd import Variable
from adacurv.torch.optim.hvp_utils import Fvp, Hvp, GNvp

###
# Build mat-vec closure helpers
###

def make_fvp_fun(closure, theta, bias_correction2=1.0):
    """
    Simply create Fvp func that doesn't require theta input so that lanczos and CG can be called
    with generic hvp funcs.
    """
    import time
    s = time.time()
    c, params = closure(theta)
    e = time.time()
    # print ("Closure time: ", (e-s))
    def f(v):
        hessp = Fvp(c, params, v)
        return hessp.data / bias_correction2
    return f

def make_fvp_fun_idx(closure, theta, params_i, params_j, bias_correction2=1.0):
    """
    Simply create Fvp func that doesn't require theta input so that lanczos and CG can be called
    with generic hvp funcs.
    """
    c, params = closure(theta, params_i, params_j)
    def f(v):
        hessp = Fvp(c, params, v)
        return hessp.data / bias_correction2
    return f

def make_hvp_fun(closure, theta, bias_correction2=1.0):
    """
    Simply create Fvp func that doesn't require theta input so that lanczos and CG can be called
    with generic hvp funcs.
    """
    c, z, params = closure(theta)
    def f(v):
        hessp = Hvp(c, params, v)
        return hessp.data / bias_correction2
    return f

def make_gnvp_fun(closure, theta, bias_correction2=1.0):
    """
    Simply create Fvp func that doesn't require theta input so that lanczos and CG can be called
    with generic hvp funcs.
    """
    c, z, params = closure(theta)
    def f(v):
        hessp = GNvp(c, z, params, v)
        return hessp.data / bias_correction2
    return f

def make_gnvp_fun_idx(closure, theta, params_i, params_j, bias_correction2=1.0):
    c, z, params = closure(theta, params_i, params_j)
    def f(v):
        hessp = GNvp(c, z, params, v)
        return hessp.data / bias_correction2
    return f

def make_fvp_obj_fun(closure, weighted_fvp_fn, ng):
    v2 = weighted_fvp_fn(ng)
    def f(p):
        pvar = Variable(p.float(), requires_grad=False)
        # vector_to_parameters(pvar, self._params_tmp)
        import time
        s = time.time()
        c, tmp_params = closure(pvar) #self._params_tmp)
        e = time.time()
        # print ("Closure time: ", (e-s))
        v1 = Fvp(c, tmp_params, ng)
        # v2 = weighted_fvp_fn(ng)
        loss = F.mse_loss(v1, v2)
        return float(loss.data)
    return f

def make_fvp_obj_fun_idx(closure, weighted_fvp_fn, ng, params_i, params_j):
    v2 = weighted_fvp_fn(ng)
    def f(p):
        pvar = Variable(p.float(), requires_grad=False)
        # vector_to_parameters(pvar, self._params_tmp)
        import time
        s = time.time()
        c, tmp_params = closure([pvar], params_i, params_j) #self._params_tmp)
        e = time.time()
        # print ("Closure time: ", (e-s))
        v1 = Fvp(c, tmp_params, ng)
        # v2 = weighted_fvp_fn(ng)
        loss = F.mse_loss(v1, v2)
        return float(loss.data)
    return f

def make_gnvp_obj_fun(closure, weighted_fvp_fn, ng):
    v2 = weighted_fvp_fn(ng)
    def f(p):
        # pvar = Variable(p.float(), requires_grad=False)
        pvar = torch.nn.Parameter(p.float()) #, requires_grad=False)
        # vector_to_parameters(pvar, self._params_tmp)
        import time
        s = time.time()
        # c, z, tmp_params = closure(pvar)
        c, z, tmp_params = closure([pvar]) #self._params_tmp)
        e = time.time()
        # print ("Closure time: ", (e-s))
        v1 = GNvp(c, z, tmp_params, ng)
        # v2 = weighted_fvp_fn(ng)
        loss = F.mse_loss(v1, v2)
        return float(loss.data)
    return f
