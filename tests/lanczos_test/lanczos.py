import numpy as np
from scipy.sparse.linalg import eigsh

np.random.seed(3)

n = 1000
# p = n
# m = 20
sqrtA = np.random.randn(n, n) - 0.5
A = np.dot(sqrtA, np.transpose(sqrtA))
# print(A)

def true_rho():
    p = n
    # print("det(A) = ", np.linalg.det(A))
    # print("Tr(A) = ", np.trace(A))
    Tr2A = np.trace(A) ** 2.0
    # print("Tr^2(A) = ", Tr2A)
    TrA2 = np.trace(A @ A)
    # print("Tr(A^2) = ", TrA2)
    # only valid with n = 2
    # print("det(A) (iden.) = ", 0.5 * ((np.trace(A) ** 2.0) - np.trace(A @ A)))
    # print("Tr(A^2) (iden.) = ", (np.trace(A) ** 2.0) - 2.0 * np.linalg.det(A))
    rho = min(((1.0 - 2.0 / p) * TrA2 + Tr2A) / ((100 + 1 - 2.0 / p) * (TrA2 - Tr2A / p)), 1.0)
    # print ("rho: ", rho)
    return rho

def approx_rho(m):
    p = n
    eA = eigsh(A, k=m, return_eigenvectors=False, which='LM') #np.linalg.eig(A)
    eA = eA[0:m]
    eTrA = np.sum(eA)
    eTr2A = eTrA ** 2.0
    eDetA = np.prod(eA)

    coef = 0.0
    for j in range(min(len(eA), m)):
        for i in range(j):
            coef += eA[j] * eA[i]

    eTrA2 = eTr2A - 2.0 * coef
    # print("eig Tr(A^2) (char.) = ", eTrA2)
    # print("eig Tr(A) = ", eTrA)
    # print("eig Tr^2(A) = ", eTr2A)
    eRho = min(((1.0 - 2.0 / p) * eTrA2 + eTr2A) / ((100 + 1 - 2.0 / p) * (eTrA2 - eTr2A / p)), 1.0)
    return eRho
    # print ("eig rho: ", eRho)


tr_rho = true_rho()
apx_rhos = []
for m in range(10, 1000, 50): #[2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90]:
    apx_rho = approx_rho(m=m)
    apx_rhos.append(apx_rho)
    # print ("Approx rho: ", approx_rho(m=m))

errors = [tr_rho - y for y in apx_rhos]
import matplotlib.pyplot as plt
plt.plot(apx_rhos)
plt.plot(errors, color='red')
plt.axhline(y=tr_rho)
plt.savefig("rhos_n.pdf")
