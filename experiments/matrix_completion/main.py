import numpy as np

rank = 10
n = 100
m = 500
U = np.random.random((n, n))
while np.linalg.det(U) <= 0:
    U = np.random.random((n, n))
V = np.random.random((m, m))
while np.linalg.det(V) <= 0:
    V = np.random.random((m, m))
P = np.zeros((n, m))
P[0:rank,0:rank] = np.eye(rank)

R = U @ P @ V
# n x m
print (R.shape)

observed = np.random.randint(0, high=2, size=R.shape)
print (observed)

k = 10
X = np.random.random((k, n))
Y = np.random.random((k, m))


def loss(X, Y, R):
    return np.linalg.norm(X.T @ Y - R) #**2 + 0.01*np.linalg.norm(X)**2 + 0.01*np.linalg.norm(Y)**2

lmda = 0.001

print ("Initial loss: ", loss(X, Y, R))

for itr in range(10):
    for u in range(n):
        y_outer = np.zeros((k, k))
        ry = np.zeros((k, 1))
        for i in range(m):
            if observed[u,i] == 1:
                y = Y[:,i][:,np.newaxis]
                y_outer += np.dot(y, y.T)
                ry += (R[u,i] * y) #[:,np.newaxis]
                # print ("ry: ", ry.shape)
                # input("")
        # print (y_outer)
        X[:,u] = (np.linalg.pinv(y_outer + lmda * np.eye(k)) @ ry)[:,0]
        # input("")

    for i in range(m):
        x_outer = np.zeros((k, k))
        rx = np.zeros((k, 1))
        for u in range(n):
            if observed[u,i] == 1:
                x = X[:,u][:,np.newaxis]
                x_outer += np.dot(x, x.T)
                rx += (R[u,i] * x) #[:,np.newaxis]

        Y[:,i] = (np.linalg.pinv(x_outer + lmda * np.eye(k)) @ rx)[:,0]

    print ("Iter: ", itr, loss(X, Y, R))
