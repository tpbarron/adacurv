import numpy as np
import time
from scipy.sparse.linalg import LinearOperator, cg

n = 10
d = 10000000

for i in range(10):
    v1 = np.random.randn(d)
    v1 /= np.linalg.norm(v1)
    v2 = np.random.randn(d)
    v2 /= np.linalg.norm(v2)

    print (np.dot(v1, v2))
input("")

A = np.random.randn(n, d)
b = np.random.randn(n, 1)

# (n x d) (d x 1) = (n x 1)

s1 = time.time()
x1, res, rank, s = np.linalg.lstsq(A, b, rcond=None)
e1 = time.time()
print ("Time: ", e1-s1)

# AA = A.T @ A
# Ab = A.T @ b
# s2 = time.time()
# x2, res, rank, s = np.linalg.lstsq(AA, Ab, rcond=None)
# # x2, info = cg(AA, Ab, maxiter=10)
# e2 = time.time()
# print ("Time: ", e2-s2)

def mv(v):
    return A.T @ (A @ v)
s3 = time.time()
lin_op = LinearOperator((d,d), matvec=mv)
x3, info = cg(lin_op, A.T @ b)
e3 = time.time()
# print (x)
print ("Time: ", e3-s3)


# print (np.linalg.norm(x1-x2))
# print (np.linalg.norm(x1-x3))


#
# s1 = time.time()
# x, res, rank, s = np.linalg.lstsq(A.T @ A, A.T @ b, rcond=None)
# e1 = time.time()
# print (x.shape)
# print ("Time: ", e1-s1)

# import torch
# A = torch.from_numpy(A)
# b = torch.from_numpy(b)
#
# s1 = time.time()
# x1, qr = torch.gels(b, A)
# e1 = time.time()
# print ("Time: ", e1-s1)
#
# s1 = time.time()
# x2, qr = torch.gels(torch.t(A) @ b, torch.t(A) @ A)
# e1 = time.time()
# print ("Time: ", e1-s1)
#
# print (np.allclose(x1.numpy(), x2.numpy()))
