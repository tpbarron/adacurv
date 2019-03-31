import numpy as np
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
sns.set_style("whitegrid", {'grid.linestyle': '--'})

plt.rc('font', family='serif')
plt.rc('text', usetex=True)

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

fig = plt.figure(figsize=(4, 3))

M = np.load('data.npy')
traj_idx = np.load('traj_idx.npy')

pca = PCA(n_components=2)
D = pca.fit_transform(M)

print (D.shape)
for i in range(10):
    print (D[i])

x, y = D.T
x *= -1.0
y *= -1.0

ti = traj_idx[40]
plt.scatter(x, y, s=4.0, color='xkcd:fuchsia')
plt.scatter(x[ti:ti+20], y[ti:ti+20], s=4.0, color='#2F4F4F')
sns.despine()

plt.xticks([-0.2, 0.0, 0.2])
# plt.show()

plt.savefig("pca.pdf")
