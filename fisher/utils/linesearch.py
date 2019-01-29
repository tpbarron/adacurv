import numpy as np
from scipy.optimize import line_search
from scipy.optimize import fmin_l_bfgs_b

def wolfe_linesearch():
    pass
    # pk = theta - theta_old
    # resp = line_search(f, fprime, theta_old.data.numpy(), pk.data.numpy(), c1=1e-5, c2=0.9, extra_condition=lambda alpha, x, f, g: alpha > 0.0 and alpha < 1.0)
    # alpha_line_search, _, _, f_new_eval, f_old_eval, slope = resp

    # pk = theta - theta_old
    # alpha_line_search, num_evals = self.backtracking_linesearch(f, fprime, theta_old.data.numpy(), pk.data.numpy())
    # if alpha_line_search is not None:
    #     print ("Alpha line search, fnew, fold, fdiff, slope: ", alpha_line_search, f_new_eval, f_old_eval, (f_old_eval-f_new_eval), slope)
    # else:
    #     print ("Alpha line search: ", alpha_line_search)

    # if alpha_line_search is None:
    #     print ("Line search 1 failed.")
    #     # import numpy as np
    #     # pk = theta - theta_old
    #     dist = 1 #float(max(torch.dist(theta, theta_old), 1))
    #     xs = np.linspace(0, dist, 500)
    #     ys = []
    #     for i in range(len(xs)):
    #         th = theta_old.data.numpy() + xs[i] * pk.data.numpy()
    #         y = f(th)
    #         ys.append(y)
    #
    #     import matplotlib.pyplot as plt
    #     plt.gcf().clear()
    #     plt.plot(xs, ys, label='y')
    #     plt.savefig("test.pdf")
    #     input("Saved")
    #     theta_old = theta_old + pk #(1-beta2) * pk
    # else:
    #     # print ("Line search succeeded.")
    #     if state['step'] % 100000000 == 0:
    #         dist = 1
    #         xs = np.linspace(0, dist, 500)
    #         ys = []
    #         for i in range(len(xs)):
    #             th = theta_old.data.numpy() + xs[i] * pk.data.numpy()
    #             y = f(th)
    #             ys.append(y)
    #
    #         import matplotlib.pyplot as plt
    #         plt.gcf().clear()
    #         plt.plot(xs, ys, label='y')
    #
    #         print ("Alpha: ", alpha_line_search)
    #         plt.plot(float(alpha_line_search), f_new_eval, 'o')
    #         plt.savefig("success.pdf")
    #         input("Saved")
    #
    #     # print ("Dist old new: ", torch.dist(theta, theta_old, p=1).data)
    #
    #     # step = self.state['step']
    #     # if step >= 1:
    #         # theta_old = theta_old + (1-float(alpha_line_search)) * pk
    #     # else:
    #     theta_old = theta_old + (1-float(alpha_line_search)) * pk

def backtracking_linesearch(f, fprime, theta_old, pk, alpha0=1.0, tau=0.001, c=1e-4, max_iters=100):
    j = 0
    alphaj = alpha0
    m = pk @ fprime(theta_old)
    t = -c * m
    print ("m: ", m)
    print ("t: ", t)
    assert m < 0.0

    fx = f(theta_old)

    itr = 0
    while itr < max_iters:
        print ("alpha * t: ", alphaj * t)
        print ("diff: ", (fx - f(theta_old + alphaj * pk)))
        cond = (fx - f(theta_old + alphaj * pk)) >= alphaj * t
        print ("cond: ", cond)
        if cond:
            return alphaj, j

        j += 1
        alphaj = tau * alphaj
        itr += 1

    return None, j

def randomized_linesearch(f, theta_old, theta, nevals=10, eps=1e-8):
    # TODO: parallelize calls to f
    alphas = np.linspace(0.0+eps, 1.0-eps, nevals)
    min_th = None
    min_f = None
    min_alpha = None
    for i in range(len(alphas)):
        alpha = alphas[i]
        th = (1 - alpha) * theta_old + alpha * theta
        f_eval = f(th)
        if min_th is None or f_eval < min_f:
            min_f = f_eval
            min_th = th
            min_alpha = alpha
    return min_th, min_f, min_alpha

def gss_linesearch(f, a, b, tol=1e-5):
    '''
    In our application:
    a = theta_old
    b = theta
    '''
    low, high = self.gss(f, a, b, tol)
    # print ("f_low: ", f(low))
    # print ("f_high: ", f(high))
    min_th = (low + high) / 2.0
    min_f = f(min_th)
    min_alpha = np.linalg.norm(min_th - a) / np.linalg.norm(b - a)
    return min_th, min_f, min_alpha

def gss(f, a, b, tol=1e-5):
    '''
    Golden section search.

    Given a function f with a single local minimum in
    the interval [a,b], gss returns a subset interval
    [c,d] that contains the minimum with d-c <= tol.

    example:
    >>> f = lambda x: (x-2)**2
    >>> a = 1
    >>> b = 5
    >>> tol = 1e-5
    >>> (c,d) = gss(f, a, b, tol)
    >>> print (c,d)
    (1.9999959837979107, 2.0000050911830893)

    Adapted from: https://en.wikipedia.org/wiki/Golden-section_search#Algorithm
    '''
    invphi = (np.sqrt(5) - 1) / 2 # 1/phi
    invphi2 = (3 - np.sqrt(5)) / 2 # 1/phi^2

    h = np.linalg.norm(b-a)
    if h <= tol:
        return (a, b)

    # required steps to achieve tolerance
    n = int(np.ceil(np.log(tol / h) / np.log(invphi)))

    c = a + invphi2 * h
    d = a + invphi * h
    yc = f(c)
    yd = f(d)

    for k in range(n-1):
        if yc < yd:
            b = d
            d = c
            yd = yc
            h = invphi * h
            c = a + invphi2 * h
            yc = f(c)
        else:
            a = c
            c = d
            yc = yd
            h = invphi * h
            d = a + invphi * h
            yd = f(d)

    if yc < yd:
        return (a, d)
    else:
        return (c, b)

def lbfgs_linesearch(f, fprime):
    resp = fmin_l_bfgs_b(f, theta.data.numpy().astype(np.float64), fprime=fprime, factr=1e7, pgtol=1e-20)
    xmin, fmin, dict = resp
    print ("fmin: ", fmin)
    print ("data: ", dict)
    input("")
    return xmin
