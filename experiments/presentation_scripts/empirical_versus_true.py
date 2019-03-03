
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
    [0.0, 1000.0]
])
def f(v):
    return 0.5 * v.transpose() @ A @ v

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

def compute_F_damped(X, Y, damping=0.001, exp=0.75):
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
    F = np.diag(np.squeeze(diag2)) + damping * np.eye(2)
    F = F**exp

    return F

def f_emp(F, v):
    # print (F)
    # print (A)
    # print (np.linalg.inv(F) @ A)
    # input("")
    return 0.5 * v.transpose() @ ((np.linalg.inv(F) * 0.1) @ A) @ v


fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
fig.set_figheight(2.75)
fig.set_figwidth(8)

print ("Condition of quadratic:", compute_condition(A))
print ("Condition damped A:", compute_condition(A + 0.1 * np.eye(A.shape[0])))

input("")
# Original
F = compute_F(X, Y)
print ("Condition of emp Fisher:", compute_condition(np.linalg.inv(F) @ A))
# print ("F: ", F)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        vec = np.array([[ X[i,j] ], [ Y[i,j] ]])
        obj = f(vec)
        Z[i,j] = obj

CS = ax1.contour(X, Y, Z)
ax1.clabel(CS, inline=1, fontsize=10)
# ax1.stitle('True curvature')
ax1.set_xticks([], [])
ax1.set_yticks([], [])

# Empirical
# F = compute_F(X, Y)
Fdamped = compute_F_damped(X, Y)
print ("Condition of emp Fisher damped:", compute_condition(np.linalg.inv(Fdamped) @ A))
# print ("F: ", F)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        vec = np.array([[ X[i,j] ], [ Y[i,j] ]])
        obj = f_emp(Fdamped, vec)
        Z[i,j] = obj

CS = ax2.contour(X, Y, Z)
ax2.clabel(CS, inline=1, fontsize=10)
# ax2.title('Conditioning by empirical curvature')
ax2.set_xticks([], [])
ax2.set_yticks([], [])

# Hessian
F = compute_F(X, Y)
# print ("F: ", F)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        vec = np.array([[ X[i,j] ], [ Y[i,j] ]])
        obj = f_quad(vec)
        Z[i,j] = obj

CS = ax3.contour(X, Y, Z)
ax3.clabel(CS, inline=1, fontsize=10)
# ax3.set_title('Condition by Hessian')
ax3.set_xticks([], [])
ax3.set_yticks([], [])



plt.tight_layout()
# plt.show()
plt.savefig("emp_vs_hess.pdf")
