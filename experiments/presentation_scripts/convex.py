
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
x = np.arange(-3, 3, delta)
y = np.arange(-2, 2, delta)
X, Y = np.meshgrid(x, y)
Z = np.zeros_like(X)


def f(x, y):
    return np.exp(x + 3*y - 0.1) + np.exp(x - 3*y - 0.1) + np.exp(-x-0.1)

def Hf(x, y):
    pass


fig, (ax1) = plt.subplots(1, 1, sharey=True)
fig.set_figheight(2.75)
fig.set_figwidth(3)


# Original
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        x, y =  X[i,j], Y[i,j]
        obj = f(x, y)
        Z[i,j] = obj

CS = ax1.contour(X, Y, Z, levels = [2, 4, 8, 16, 32, 64, 128, 256])
ax1.clabel(CS, fmt='%2.1f', inline=1, fontsize=10)
# ax1.stitle('True curvature')
ax1.set_xticks([], [])
ax1.set_yticks([], [])


plt.tight_layout()
# plt.show()
plt.savefig("convex.pdf")
