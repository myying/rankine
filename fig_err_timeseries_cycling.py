#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from config import *

cases = ('NoDA_s1_1', 'EnSRF_s1_1', 'EnSRF_s3_1', 'EnSRF_s3_3')
casename = ('NoDA', 'EnSRF', 'EnSRF_MSA_3', 'EnSRF_MSA-O_3')
colors = ([.7, .7, .7], [0, 0, 0], [0, .7, .9], [.8, .3, .3])
linewidths = (4, 3, 2, 2)
nens = (30, 30, 25, 25)
expname = ('cycling/perfect_model/type2', 'cycling/perfect_model/type2', 'cycling/perfect_model/type2', 'cycling/imperfect_model/type2')
scenario = ('Lsprd3/struct_perturb0/phase1.0', 'Lsprd3/struct_perturb0/phase0.0', 'Lsprd3/struct_perturb1/phase1.0', 'Lsprd3/struct_perturb0/phase1.0')
nreal = (100, 100, 100, 100)
ncycle = int(nt/obs_t_intv)

ts = np.zeros(nt*2+1)
ts[0:nt*2+1:2] = np.arange(nt+1)
ts[1:nt*2:2] = np.arange(nt)

fig, ax = plt.subplots(4, len(scenario), figsize=(14, 12))
for j in range(len(scenario)):
    for c in range(len(cases)):
        rmse = np.zeros((nreal[j], nt*2+1, 4))
        rmse_fcst = np.zeros((nreal[j], ncycle, nt+1, 4))
        r1 = 0
        for r in range(nreal[j]):
            rmse[r1, 0:nt*2+1:2, 0] = np.load('output/'+expname[j]+'/{:04d}/{}/{}.npy'.format(r+1, scenario[j], cases[c]))[nens[c], 0, 0, 0:nt+1]
            rmse[r1, 0:nt*2+1:2, 1] = np.mean(np.load('output/'+expname[j]+'/{:04d}/{}/{}.npy'.format(r+1, scenario[j], cases[c]))[0:nens[c], 1, 0, 0:nt+1], axis=0)*9
            rmse[r1, 0:nt*2+1:2, 2] = np.mean(np.load('output/'+expname[j]+'/{:04d}/{}/{}.npy'.format(r+1, scenario[j], cases[c]))[0:nens[c], 2, 0, 0:nt+1], axis=0)
            rmse[r1, 0:nt*2+1:2, 3] = np.mean(np.load('output/'+expname[j]+'/{:04d}/{}/{}.npy'.format(r+1, scenario[j], cases[c]))[0:nens[c], 3, 0, 0:nt+1], axis=0)*9
            rmse[r1, 1:nt*2:2, 0] = np.load('output/'+expname[j]+'/{:04d}/{}/{}.npy'.format(r+1, scenario[j], cases[c]))[nens[c], 0, 1, 0:nt]
            rmse[r1, 1:nt*2:2, 1] = np.mean(np.load('output/'+expname[j]+'/{:04d}/{}/{}.npy'.format(r+1, scenario[j], cases[c]))[0:nens[c], 1, 1, 0:nt], axis=0)*9
            rmse[r1, 1:nt*2:2, 2] = np.mean(np.load('output/'+expname[j]+'/{:04d}/{}/{}.npy'.format(r+1, scenario[j], cases[c]))[0:nens[c], 2, 1, 0:nt], axis=0)
            rmse[r1, 1:nt*2:2, 3] = np.mean(np.load('output/'+expname[j]+'/{:04d}/{}/{}.npy'.format(r+1, scenario[j], cases[c]))[0:nens[c], 3, 1, 0:nt], axis=0)*9
            for cycle in range(2, ncycle):
                rmse_fcst[r1, cycle, :, 0] = np.load('output/'+expname[j]+'/{:04d}/{}/{}_fcst.npy'.format(r+1, scenario[j], cases[c]))[nens[c], 0, cycle, :]
                rmse_fcst[r1, cycle, :, 1] = np.mean(np.load('output/'+expname[j]+'/{:04d}/{}/{}_fcst.npy'.format(r+1, scenario[j], cases[c]))[0:nens[c], 1, cycle, :], axis=0)*9
                rmse_fcst[r1, cycle, :, 2] = np.mean(np.load('output/'+expname[j]+'/{:04d}/{}/{}_fcst.npy'.format(r+1, scenario[j], cases[c]))[0:nens[c], 2, cycle, :], axis=0)
                rmse_fcst[r1, cycle, :, 3] = np.mean(np.load('output/'+expname[j]+'/{:04d}/{}/{}_fcst.npy'.format(r+1, scenario[j], cases[c]))[0:nens[c], 3, cycle, :], axis=0)*9
                rmse_fcst[r1, cycle, cycle*obs_t_intv, :] = rmse[r1, 2*cycle*obs_t_intv, :]
            r1 += 1
        for i in range(4):
            mean_err_ts = np.mean(rmse[0:r1, :, i], axis=0)
            ax[i, j].plot(ts, mean_err_ts, color=colors[c], linewidth=linewidths[c], label=casename[c])
            if i==3 and j==3:
                ax[i, j].legend(fontsize=12)
            if c>0:
                for cycle in range(2, ncycle):
                    ax[i, j].plot(np.arange(cycle*obs_t_intv, nt+1), np.mean(rmse_fcst[0:r1, cycle, cycle*obs_t_intv:nt+1, i], axis=0), linestyle=':', color=colors[c], linewidth=2)
ymax = (6, 60, 20, 12)
for j in range(len(scenario)):
    for i in range(4):
        ax[i, j].set_xlim([0, nt])
        ax[i, j].set_ylim([0, ymax[i]])
        ax[i, j].set_xticks([0, 3, 6, 9, 12])
        ax[i, j].tick_params(labelsize=12)
plt.savefig('out.pdf')
