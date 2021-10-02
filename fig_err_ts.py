#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from config import *

# cases = ('NoDA_s1_1', 'EnSRF_s1_1', 'EnSRF_s2_1', 'EnSRF_s2_2', 'EnSRF_s3_1', 'EnSRF_s3_3', 'EnSRF_s4_1', 'EnSRF_s4_4')
# nens = (30, 30, 28, 28, 25, 25, 22, 22)
cases = ('NoDA_s1_1', 'EnSRF_s1_1', 'EnSRF_s3_1', 'EnSRF_s3_3')
colors = ([.7, .7, .7], 'k', [0, 1, 0], [.8, .4, .4])
nens = (30, 30, 25, 25)
expname = ('cycling/perfect_model/type2', 'cycling/perfect_model/type2', 'cycling/imperfect_model/type2')
scenario = ('Lsprd3/phase1.0', 'Lsprd3/phase0.0', 'Lsprd3/phase1.0')
nreal = (100, 100, 100)

ts = np.zeros(nt*2)
ts[0::2] = np.arange(nt)
ts[1::2] = np.arange(nt)

fig, ax = plt.subplots(4, 3, figsize=(12, 12))
k = 0  ##0=prior, 1=posterior
for j in range(3):
    for c in range(len(cases)):
        rmse = np.zeros((nreal[j], nt*2, 4))
        r1 = 0
        for r in range(nreal[j]):
            for l in (0, 1):
                rmse[r1, l::2, 0] = np.load('output/'+expname[j]+'/{:04d}/{}/{}.npy'.format(r+1, scenario[j], cases[c]))[-1, 0, l, :]
                rmse[r1, l::2, 1] = np.mean(np.load('output/'+expname[j]+'/{:04d}/{}/{}.npy'.format(r+1, scenario[j], cases[c]))[0:nens[c], 1, l, :], axis=0)*9
                rmse[r1, l::2, 2] = np.mean(np.load('output/'+expname[j]+'/{:04d}/{}/{}.npy'.format(r+1, scenario[j], cases[c]))[0:nens[c], 2, l, :], axis=0)
                rmse[r1, l::2, 3] = np.mean(np.load('output/'+expname[j]+'/{:04d}/{}/{}.npy'.format(r+1, scenario[j], cases[c]))[0:nens[c], 3, l, :], axis=0)*9
            r1 += 1
        for i in range(4):
            mean_err_ts = np.mean(rmse[0:r1, :, i], axis=0)
            ax[i, j].plot(ts, mean_err_ts, color=colors[c])
# ymax = (1.8, 45, 8, 10)
# for j in range(1):
    # for i in range(4):
        # ax[i, j].grid()
        # ax[i, j].set_xlim([0, len(Lsprd)])
        # ax[i, j].set_ylim([0, ymax[i]])
        # ax[i, j].set_xticks([0, 1, 2])
        # ax[i, j].set_xticklabels([])
        # ax[i, j].set_axisbelow(True)
        # ax[i, j].tick_params(labelsize=12)
plt.savefig('out.pdf')
