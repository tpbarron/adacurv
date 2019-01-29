import matplotlib.pyplot as plt
import numpy as np

ind = 2

base_path = 'results/inverted_pend_lin_test_exp'
# Standard NPG
data1 = np.loadtxt(base_path+"1/results.txt", skiprows=2)
y1 = data1[:,ind]

# Standard NPG own
data2 = np.loadtxt(base_path+"2/results.txt", skiprows=2)
y2 = data2[:,ind]
data8 = np.loadtxt(base_path+"8/results.txt", skiprows=2)
y8 = data8[:,ind]
data9 = np.loadtxt(base_path+"9/results.txt", skiprows=2)
y9 = data9[:,ind]

# adaptive NGD beta(0.1)
data3 = np.loadtxt(base_path+"3/results.txt", skiprows=2)
y3 = data3[:,ind]

# adaptive NGD beta(0.2)
data4 = np.loadtxt(base_path+"4/results.txt", skiprows=2)
y4 = data4[:,ind]

# adaptive NGD beta(0.3)
data5 = np.loadtxt(base_path+"5/results.txt", skiprows=2)
y5 = data5[:,ind]

# adaptive NGD shrunk beta(0.1)
data6 = np.loadtxt(base_path+"6/results.txt", skiprows=2)
y6 = data6[:,ind]
data7 = np.loadtxt(base_path+"7/results.txt", skiprows=2)
y7 = data7[:,ind]

# adaptive NGD beta (0.1) lr 0.01
data10 = np.loadtxt(base_path+"10/results.txt", skiprows=2)
y10 = data10[:,ind]
# adaptive NGD beta (0.25) lr 0.01
data11 = np.loadtxt(base_path+"11/results.txt", skiprows=2)
y11 = data11[:,ind]

# adaptive NGD beta (0.1) lr 0.005
data12 = np.loadtxt(base_path+"12/results.txt", skiprows=2)
y12 = data12[:,ind]
# NGD lr 0.005
data13 = np.loadtxt(base_path+"13/results.txt", skiprows=2)
y13 = data13[:,ind]

# adaptive NGD beta (0.1) lr 0.005
data12 = np.loadtxt(base_path+"14/results.txt", skiprows=2)
y12 = data12[:,ind]
# NGD lr 0.005
data13 = np.loadtxt(base_path+"15/results.txt", skiprows=2)
y13 = data13[:,ind]

# plt.plot(np.arange(len(y1)), y1, label='npg')
# plt.plot(np.arange(len(y2)), y2, label='npg opt')
# plt.plot(np.arange(len(y8)), y8, label='npg opt (seed 13)')
# plt.plot(np.arange(len(y9)), y9, label='npg opt (seed 17)')
# plt.plot(np.arange(len(y3)), y3, label='anpg 0.1 lr 0.005')

# plt.plot(np.arange(len(y4)), y4, label='anpg 0.2 lr 0.005')
# plt.plot(np.arange(len(y5)), y5, label='anpg 0.3 lr 0.005')

# plt.plot(np.arange(len(y6)), y6, label='anpg shrunk 0.1 lr 0.005 (seed 13)')
# plt.plot(np.arange(len(y7)), y7, label='anpg shrunk 0.1 lr 0.005 (seed 17)')

# plt.plot(np.arange(len(y10)), y10, label='anpg 0.1 lr 0.01')
# plt.plot(np.arange(len(y10)), y11, label='anpg 0.25 lr 0.01')

plt.plot(np.arange(len(y12)), y12, label='anpg 0.1 lr 0.005')
plt.plot(np.arange(len(y13)), y13, label='npg lr 0.005')

plt.legend()
plt.tight_layout()
plt.savefig("npg_inv_pend_lr0.005_nodecay.pdf")
