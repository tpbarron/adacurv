import matplotlib.pyplot as plt
import numpy as np

ind = 2

# Standard NPG
data1 = np.loadtxt("lunar_lander_nn_ngd_cg_exp1/results.txt", skiprows=2)
y1 = data1[:,ind]

# Adaptive beta(0.1, 0.1)
data2 = np.loadtxt("lunar_lander_nn_adaptive_ngd_cg_exp2/results.txt", skiprows=2)
y2 = data2[:,ind]

# Adaptive beta(0.5, 0.5)
data3 = np.loadtxt("lunar_lander_nn_adaptive_ngd_cg_exp3/results.txt", skiprows=2)
y3 = data3[:,ind]

# Adaptive beta(0.9, 0.9)
data4 = np.loadtxt("lunar_lander_nn_adaptive_ngd_cg_exp4/results.txt", skiprows=2)
y4 = data4[:,ind]

# Adaptive shrunk beta(0.9, 0.9)
data5 = np.loadtxt("lunar_lander_nn_adaptive_ngd_cg_shrunk_exp5/results.txt", skiprows=2)
y5 = data5[:,ind]

# Adaptive beta (0.9, 0.9) lr 0.0005
data6 = np.loadtxt("lunar_lander_nn_adaptive_ngd_cg_exp6/results.txt", skiprows=2)
y6 = data6[:,ind]

plt.plot(np.arange(len(y1)), y1, label='npg')
plt.plot(np.arange(len(y2)), y2, label='anpg 0.1')
plt.plot(np.arange(len(y3)), y3, label='anpg 0.5')
# plt.plot(np.arange(len(y4)), y4, label='anpg 0.9')
# plt.plot(np.arange(len(y5)), y5, label='anpg shrunk 0.9')
plt.plot(np.arange(len(y6)), y6, label='anpg 0.9 lr 0.0005')

plt.legend()
plt.tight_layout()
plt.savefig("npg.pdf")
