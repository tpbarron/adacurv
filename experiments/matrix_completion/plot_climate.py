import numpy as np
from numpy.polynomial.polynomial import polyfit

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import numpy as np
import seaborn as sns
import os

sns.set()
sns.set_style("whitegrid", {'grid.linestyle': '--'})

plt.rc('font', family='serif')
# plt.rc('text', usetex=True)
# pgf_with_rc_fonts = {"pgf.texsystem": "pdflatex"}
# import matplotlib
# matplotlib.rcParams.update(pgf_with_rc_fonts)

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

def apply_denormalization(P, norm_offset, norm_scale):
    P_denorm = np.copy(P)
    # For every row in matrix
    for i in range(P_denorm.shape[0]):
        # Get ids for each month
        # Normalized over months
        for m in range(12):
            idxs = np.arange(m, P_denorm.shape[1], 12)

            norm_scl = norm_scale[i][m]
            norm_off = norm_offset[i][m]

            P_denorm[i,idxs] *= norm_scl
            P_denorm[i,idxs] += norm_off
    return P_denorm

dimension_map = {
    'historical': [0],
    'rcp_toa': [1, 2, 3, 4, 5, 6],
    'rcp_tos': [7, 8, 9, 10, 11, 12],
    'rcp_zosga': [13, 14, 15, 16, 17],
    'noaa_billion_cost': [18],
    'noaa_billion_deaths': [19]
}

def plot_dimension(P, M, W, key):
    dims = dimension_map[key]

    for d in dims:
        plt.plot(P[d,:])
        plt.title(key)
    plt.show()

def plot_pred_anomaly(P, M, W, rcp, r, gn):
    fig = plt.figure(figsize=(4, 3))

    true_idx = np.nonzero(W[0])
    plt.scatter(np.arange(1900, 2019.01, 1.0/12), M[0][true_idx], color='blue', s=2)
    # plt.plot(np.arange(1900, 2019.01, 1.0/12), M[0][true_idx], '.')
    pred_idx = np.nonzero(W[0] == 0)
    plt.scatter(np.arange(2019, 2100.9, 1.0/12), P[0][pred_idx], color='orange', s=2)
    # plt.plot(np.arange(2019, 2100.9, 1.0/12), P[0][pred_idx], '.')

    plt.ylabel('Mid-century Temp. Anomaly')
    plt.xlabel('Year')

    sns.despine()

    plt.tight_layout()

    plt.savefig('models/figs/pred_anomaly_'+rcp+'_rank'+str(r)+'_'+gn+'.pdf')
    # plt.show()


def plot_pred_cost(P, M, W, rcp, r, gn):
    fig = plt.figure(figsize=(4, 3))

    i = 13
    true_idx = np.nonzero(W[i])
    x = np.arange(1900, 2019.01, 1.0/12)[true_idx]
    y = M[i][true_idx]
    # print ("y trues: ", y)
    plt.scatter(x, y, color='blue', s=2)

    pred_idx = np.nonzero(W[i] == 0)
    x = np.arange(1900, 2101.01, 1.0/12)[pred_idx]
    y = P[i][pred_idx]

    x_valid = x > 2018
    x = x[x_valid]
    y = y[x_valid]
    plt.scatter(x, y, color='orange', s=2)

    # Fit with polyfit
    b, m1, m2 = polyfit(x, y, 2)
    plt.plot(x, b + m1 * x + m2 * x **2.0, '-')

    # plt.gca().set_yscale('log')

    plt.ylabel('Est. Financial Cost (Billions)')
    plt.xlabel('Year')

    sns.despine()

    plt.tight_layout()

    plt.savefig('models/figs/pred_cost_'+rcp+'_rank'+str(r)+'_'+gn+'.pdf')
    # plt.show()

def plot_pred_deaths(P, M, W, rcp, r, gn):
    fig = plt.figure(figsize=(4, 3))

    i = 14
    true_idx = np.nonzero(W[i])
    x = np.arange(1900, 2019.01, 1.0/12)[true_idx]
    y = M[i][true_idx]
    print ("y trues: ", y)
    plt.scatter(x, y, color='blue', s=2)

    pred_idx = np.nonzero(W[i] == 0)
    x = np.arange(1900, 2101.01, 1.0/12)[pred_idx]
    y = P[i][pred_idx]

    x_valid = x > 2018
    x = x[x_valid]
    y = y[x_valid]
    plt.scatter(x, y, color='orange', s=2)

    # Fit with polyfit
    b, m1, m2 = polyfit(x, y, 2)
    plt.plot(x, b + m1 * x + m2 * x **2.0, '-')

    # plt.gca().set_yscale('log')

    plt.ylabel('Est. Deaths')
    plt.xlabel('Year')

    sns.despine()

    plt.tight_layout()

    plt.savefig('models/figs/pred_deaths_'+rcp+'_rank'+str(r)+'_'+gn+'.pdf')

# def plot_dimension(P, M, W, key):
#     # dims = dimension_map[key]
#
#     x, y = np.arange(1900, 2101, 1.0/12), P[19]
#
#     # Fit with polyfit
#     b, m = polyfit(x, y, 1)
#
#     plt.plot(x, y, '.')
#     plt.plot(x, b + m * x, '-')
#     # for d in dims:
#     #     plt.plot(P[d,:])
#     #     plt.title(key)
#     plt.show()
#

if __name__ == "__main__":

    rcp = 'rcp45'
    r = 5
    gn = 'gn'
    P = np.load('models/P_'+rcp+'_rank'+str(r)+'_'+gn+'.npy')
    M = np.load('data/climate_data/matrices/M_1900_2101_'+rcp+'.npy')
    W = np.load('data/climate_data/matrices/W_1900_2101_'+rcp+'.npy')

    norm_offset = np.load('data/climate_data/matrices/norm_offset_1900_2101_'+rcp+'.npy')
    norm_scale = np.load('data/climate_data/matrices/norm_scale_1900_2101_'+rcp+'.npy')

    P_denorm = apply_denormalization(P, norm_offset, norm_scale)
    M_denorm = apply_denormalization(M, norm_offset, norm_scale)

    plot_pred_anomaly(P_denorm, M_denorm, W, rcp, r, gn)
    plot_pred_cost(P_denorm, M_denorm, W, rcp, r, gn)
    plot_pred_deaths(P_denorm, M_denorm, W, rcp, r, gn)

    # for k in dimension_map.keys():
    #     plot_dimension(P_denorm, M_denorm, W, k)
    # plot_dimension(P_denorm, M_denorm, W, '') #k)
