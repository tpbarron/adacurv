
import numpy as np

import matplotlib
import numpy as np
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
sns.set_style("white")
sns.set_context("paper")

plt.rc('font', family='serif')
plt.rc('text', usetex=True)

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

# plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
# plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
# plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
# plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
# plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
# plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
# plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


fig = plt.figure(figsize=(4, 3))

xs = np.linspace(0, 1.0, 100)
y1 = 1.5 * (xs)**2
y2 = 1.0 * (xs-1)**2
y3 = y1 + y2

ymin = np.argmin(y3)

plt.plot(xs, y1, color='orange', label='Squared Bias')
plt.plot(xs, y2, color='blue', label='Variance')
plt.plot(xs, y3, color='black', label='MSE')

plt.plot(xs[ymin], y3[ymin], color='red', marker='.', ms=10)
plt.xlabel(r"Shrinkage ($\rho$)")
plt.xticks([0, 1])
plt.yticks([])

plt.legend()
plt.tight_layout()

sns.despine()
plt.savefig("bias_v_variance.pdf")
# plt.show()
