import numpy as np

def perform_induction_step(M):
    d = M.shape[0]
    dnew = d+1
    Mnew = np.eye(dnew)
    Mnew[1:,1:] = M

    v = np.random.randn(dnew)[:,np.newaxis]
    v /= np.linalg.norm(v)

    H = np.eye(dnew) - 2.0 * v @ v.T
    Mnew = H @ Mnew
    return Mnew

def random_rot(dim=10):
    th = np.random.uniform(low=0.0, high=2.0*np.pi)
    M = np.array([[np.cos(th), -np.sin(th)],
                  [np.sin(th), np.cos(th)]])
    while M.shape[0] < dim:
        print (M)
        input("")
        M = perform_induction_step(M)

    print (M)


if __name__ == '__main__':
    random_rot(dim=5)
