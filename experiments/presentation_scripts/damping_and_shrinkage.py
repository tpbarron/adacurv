
import numpy as np

import matplotlib
import numpy as np
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
sns.set_style("white")

# matplotlib.rcParams['xtick.direction'] = 'out'
# matplotlib.rcParams['ytick.direction'] = 'out'

delta = 0.025
x = np.arange(-3.0, 3.0, delta)
y = np.arange(-3.0, 3.0, delta)
X, Y = np.meshgrid(x, y)
Z = np.zeros_like(X)


def compute_condition(M):
    eigs = np.linalg.eigvals(M)
    return np.max(eigs) / np.min(eigs)

# A = np.random.random((2, 2))
# A = A @ A.T
A = np.array([
    [1.0, 0.0],
    [0.0, 10.0]
])
def f(v, damping=0.0):
    return 0.5 * v.transpose() @ (A + damping * np.eye(A.shape[0])) @ v

def f_shrunk(C, v):
    return 0.5 * v.transpose() @ C @ v

def f_quad(v):
    return 0.5 * v.transpose() @ (np.linalg.inv(A) @ A) @ v

def compute_F(X, Y):
    n = 0
    g_hat = np.zeros((2, 1))
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            vec = np.array([[ X[i,j] ], [ Y[i,j] ]])
            g = A @ vec
            g_hat += g
            n += 1
    g_hat /= n
    diag2 = g_hat * g_hat
    F = np.diag(np.squeeze(diag2))
    return F

def f_emp(F, v):
    # print (F)
    # print (A)
    # print (np.linalg.inv(F) @ A)
    # input("")
    return 0.5 * v.transpose() @ ((np.linalg.inv(F) * 0.1) @ A) @ v


fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
fig.set_figheight(2.75)
fig.set_figwidth(5.5)

print ("Condition of quadratic:", compute_condition(A))
print ("Condition damped A:", compute_condition(A + 0.1 * np.eye(A.shape[0])))

# print ("F: ", F)
# for i in range(X.shape[0]):
#     for j in range(X.shape[1]):
#         vec = np.array([[ X[i,j] ], [ Y[i,j] ]])
#         obj = f(vec)
#         Z[i,j] = obj
#
# CS = ax1.contour(X, Y, Z)
# ax1.clabel(CS, inline=1, fontsize=10)
# # ax1.stitle('True curvature')
# ax1.set_xticks([], [])
# ax1.set_yticks([], [])

# print ("F: ", F)
# for i in range(X.shape[0]):
#     for j in range(X.shape[1]):
#         vec = np.array([[ X[i,j] ], [ Y[i,j] ]])
#         obj = f(vec, 0.1)
#         Z[i,j] = obj
#
# CS = ax2.contour(X, Y, Z)
# ax2.clabel(CS, inline=1, fontsize=10)
# # ax2.title('Conditioning by empirical curvature')
# ax2.set_xticks([], [])
# ax2.set_yticks([], [])


n = 10
p = 2
C = np.zeros_like(A)
for i in range(n):
    e1 = 1 + np.random.randn()
    e2 = 10 + np.random.randn()
    V = np.array([[e1, 0], [0, e2]])
    C += V
C /= n

print ("Condition C no shrunk:", compute_condition(C))

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        vec = np.array([[ X[i,j] ], [ Y[i,j] ]])
        obj = f_shrunk(C, vec)
        Z[i,j] = obj

CS = ax1.contour(X, Y, Z)
ax1.clabel(CS, inline=1, fontsize=10)
# ax1.stitle('True curvature')
ax1.set_xticks([], [])
ax1.set_yticks([], [])

# print (C)
# input("")

trC = np.trace(C)
trC2 = np.trace(C @ C)

tr2C = trC ** 2.0
Dt = trC / p * np.eye(2)

numer = (1.0-2.0) / p * trC2 + tr2C
denom = (n + 1.0 - 2.0) / p  * (trC2 +-tr2C / p)
rho = np.minimum(  numer / denom  , 1.0)
# print (rho)
Cs = (1-rho) * C + rho * Dt
# print (Cs)

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        vec = np.array([[ X[i,j] ], [ Y[i,j] ]])
        obj = f_shrunk(Cs, vec)
        Z[i,j] = obj

CS = ax2.contour(X, Y, Z)
ax2.clabel(CS, inline=1, fontsize=10)
# ax2.title('Conditioning by empirical curvature')
ax2.set_xticks([], [])
ax2.set_yticks([], [])


print ("Condition Ashrunk:", compute_condition(Cs))


plt.tight_layout()
# plt.show()
plt.savefig("damping.pdf")
