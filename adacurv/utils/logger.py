
import os
import pickle

class DictLogger:
    """
    Log a set of variables 'on demand' in dictionary form, e.g.:

    data: {
        hyperparameters: {
            lr: ...,
            beta1: ...,
            ...
        }
        iterations: {
            0: {
                params: ...,
                gradient: ...,
                cg: {
                    0: [res, direction, cg_delta],
                    1: [res, direction, cg_delta],
                    ...
                },
                loss: ...
            },
            1: {
                params: ...,
                gradient: ...,
                cg: {
                    0: [fvp, residual, direction, cg_delta],
                    1: [fvp, residual, direction, cg_delta],
                    ...
                },
                loss: ...
            },
            ...
        }
    }
    """

    def __init__(self, log_dir='/tmp/adacurv'):
        self.log_dir = log_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        self.iteration = 0
        self.data = {}
        self.data['iterations'] = {}
        self.data['hyperparameters'] = {}

    def next_iteration(self):
        self.iteration += 1

    def log_hyperparam(self, key, val):
        self.data['hyperparameters'][key] = val

    def log_hyperparam_dict(self, param_dict):
        for key, val in param_dict.items():
            self.log_hyperparam(key, val)

    def log_kv(self, key, val):
        if self.iteration not in self.data['iterations']:
            self.data['iterations'][self.iteration] = {}
        if key in self.data['iterations'][self.iteration]:
            self.data['iterations'][self.iteration][key].append(val)
        else:
            if type(val) is list:
                self.data['iterations'][self.iteration][key] = val
            else:
                self.data['iterations'][self.iteration][key] = [val]

    def save_log(self):
        fpath = os.path.join(self.log_dir, 'log.pkl')
        with open(fpath, 'wb') as f:
            pickle.dump(self.data, f)
        # print (self.data['iterations'][0]['cg'])
        # input("")


def logs_have_equivalent_hyperparams(log1, log2):
    hyperparameters1 = log1['hyperparameters']
    hyperparameters2 = log2['hyperparameters']
    keys = set(list(hyperparameters1.keys()) + list(hyperparameters2.keys()))
    for k in keys:
        if k not in hyperparameters2:
            print ("hyperparameters2 does not have key: ", k)
            return False
        if k not in hyperparameters1:
            print ("hyperparameters1 does not have key: ", k)
            return False
        v1 = hyperparameters1[k]
        v2 = hyperparameters2[k]
        if v1 != v2:
            print ("key exists but value different.")
            print ("v1: ", v1)
            print ("v2: ", v2)
            return False
    return True

import numpy as np

def _equivalent_params_pre(params_pre1, params_pre2):
    params_pre1.sort(key=lambda x: np.prod(x.shape))
    params_pre2.sort(key=lambda x: np.prod(x.shape))
    params_pre2[1] = params_pre2[1].T
    for p1, p2 in zip(params_pre1, params_pre2):
        print ("params_pre close: ", np.allclose(p1, p2))

def _equivalent_params_old_pre(params_old_pre1, params_old_pre2):
    params_old_pre1.sort(key=lambda x: np.prod(x.shape))
    params_old_pre2.sort(key=lambda x: np.prod(x.shape))
    params_old_pre2[1] = params_old_pre2[1].T
    for p1, p2 in zip(params_old_pre1, params_old_pre2):
        print ("params_old_pre close: ", np.allclose(p1, p2))

def _equivalent_gradients(grads1, grads2):
    grads1.sort(key=lambda x: np.prod(x.shape))
    grads2.sort(key=lambda x: np.prod(x.shape))
    grads2[1] = grads2[1].T
    for p1, p2 in zip(grads1, grads2):
        print ("grads close: ", np.allclose(p1, p2, atol=1e-5))

def _equivalent_m(m1, m2):
    m1.sort(key=lambda x: np.prod(x.shape))
    m2.sort(key=lambda x: np.prod(x.shape))
    m2[1] = m2[1].T

    print (m2[0], m2[1])
    # m2_flat = np.concatenate([m2i.reshape(-1) for m2i in m2])
    print (m1[0][0:20])
    print (m1[0][-20:])
    # print ("m close: ", np.allclose(m1, m2_flat, atol=1e-5))
    #
    # for p1, p2 in zip(m1, m2):
    #     print (p1.shape, p2.shape)
    #     print ("m close: ", np.allclose(p1, p2, atol=1e-5))

def logs_have_equivalent_iteration(log1, log2):
    iterations1 = log1['iterations']
    iterations2 = log2['iterations']
    for i in range(1):
        itr1 = iterations1[0]
        itr2 = iterations2[0]
        _equivalent_params_pre(itr1['params_pre'], itr2['params_pre'])
        _equivalent_params_old_pre(itr1['params_old_pre'], itr2['params_old_pre'])
        _equivalent_gradients(itr1['gradient'], itr2['gradient'])
        _equivalent_m(itr1['m'], itr2['m'])


def find_where_not_equal(logpath1, logpath2):
    """
    Find where this and other differ.
    """
    # Open logs
    log1, log2 = None, None
    with open(logpath1, 'rb') as logf1:
        log1 = pickle.load(logf1)
    if not log1:
        raise FileNotFoundError("Log1 not found")
    with open(logpath2, 'rb') as logf2:
        log2 = pickle.load(logf2)
    if not log2:
        raise FileNotFoundError("Log2 not found")

    # Check hyperparameter equivalence
    print ("Eq hyperparams: ", logs_have_equivalent_hyperparams(log1, log2))
    logs_have_equivalent_iteration(log1, log2)
