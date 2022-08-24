#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from config import *

cases = ('NoDA_s1_1', 'EnSRF_s1_1', 'EnSRF_s2_1', 'EnSRF_s2_2', 'EnSRF_s3_1', 'EnSRF_s3_3', 'EnSRF_s4_1', 'EnSRF_s4_4')
nens = (30, 30, 28, 28, 25, 25, 22, 22)
expname = ('cycling/perfect_model/type2', 'cycling/perfect_model/type2', 'cycling/perfect_model/type2', 'cycling/imperfect_model/type2')
scenario = ('Lsprd3/struct_perturb0/phase1.0', 'Lsprd3/struct_perturb0/phase0.0', 'Lsprd3/struct_perturb1/phase1.0', 'Lsprd3/struct_perturb0/phase1.0')
nreal = (100, 100, 100, 100)

fig, ax = plt.subplots(4, len(scenario), figsize=(12, 12))
for j in range(len(scenario)):
    for l in (0, 1):  ##0=prior, 1=posterior (forecast)
        for c in range(len(cases)):
            rmse = np.zeros((nreal[j], 4))
            r1 = 0
            for r in range(nreal[j]):
                if l==0:  ##analysis errors
                    rmse[r1, 0] = np.mean(np.load('output/'+expname[j]+'/{:04d}/{}/{}.npy'.format(r+1, scenario[j], cases[c]))[nens[c], 0, 1, 3:10:3])
                    rmse[r1, 1] = np.mean(np.load('output/'+expname[j]+'/{:04d}/{}/{}.npy'.format(r+1, scenario[j], cases[c]))[0:nens[c], 1, 1, 3:10:3])*9
                    rmse[r1, 2] = np.mean(np.load('output/'+expname[j]+'/{:04d}/{}/{}.npy'.format(r+1, scenario[j], cases[c]))[0:nens[c], 2, 1, 3:10:3])
                    rmse[r1, 3] = np.mean(np.load('output/'+expname[j]+'/{:04d}/{}/{}.npy'.format(r+1, scenario[j], cases[c]))[0:nens[c], 3, 1, 3:10:3])*9
                if l==1:  ##forecast errors at end of period
                    rmse[r1, 0] = np.load('output/'+expname[j]+'/{:04d}/{}/{}.npy'.format(r+1, scenario[j], cases[c]))[nens[c], 0, 0, -1]
                    rmse[r1, 1] = np.mean(np.load('output/'+expname[j]+'/{:04d}/{}/{}.npy'.format(r+1, scenario[j], cases[c]))[0:nens[c], 1, 0, -1])*9
                    rmse[r1, 2] = np.mean(np.load('output/'+expname[j]+'/{:04d}/{}/{}.npy'.format(r+1, scenario[j], cases[c]))[0:nens[c], 2, 0, -1])
                    rmse[r1, 3] = np.mean(np.load('output/'+expname[j]+'/{:04d}/{}/{}.npy'.format(r+1, scenario[j], cases[c]))[0:nens[c], 3, 0, -1])*9
                    if c>0:
                        for cycle in (1, 2):
                            rmse[r1, 0] += np.load('output/'+expname[j]+'/{:04d}/{}/{}_fcst.npy'.format(r+1, scenario[j], cases[c]))[nens[c], 0, cycle, -1]
                            rmse[r1, 1] += np.mean(np.load('output/'+expname[j]+'/{:04d}/{}/{}_fcst.npy'.format(r+1, scenario[j], cases[c]))[0:nens[c], 1, cycle, -1])*9
                            rmse[r1, 2] += np.mean(np.load('output/'+expname[j]+'/{:04d}/{}/{}_fcst.npy'.format(r+1, scenario[j], cases[c]))[0:nens[c], 2, cycle, -1])
                            rmse[r1, 3] += np.mean(np.load('output/'+expname[j]+'/{:04d}/{}/{}_fcst.npy'.format(r+1, scenario[j], cases[c]))[0:nens[c], 3, cycle, -1])*9
                        rmse[r1, :] /= 3
                r1 += 1
            # print(expname[j],scenario[j],np.where(np.isnan(rmse[:,:])))  ##check for nan and rerun realization then come back

            for i in range(4):
                x = l + c*0.1 + 0.1
                q3, q1 = np.percentile(rmse[0:r1, i], [75, 25])
                median = np.median(rmse[0:r1, i])
                if c==0:
                    fc = [.7, .7, .7]
                    ec = [0, 0, 0]
                if c==1:
                    fc = [1, 1, 1]
                    ec = [0, 0, 0]
                if c in (2, 4, 6):
                    fc = [0, .7, .9]
                    ec = None
                if c in (3, 5, 7):
                    fc = [.8, .3, .3]
                    ec = None
                ax[i, j].add_patch(Polygon([(x-0.04,q1), (x-0.04,q3), (x+0.04,q3), (x+0.04,q1)], facecolor=fc, ec=ec))
                ax[i, j].plot(x, median, marker='.', color='black')
ymax = (6.5, 70, 18, 10)
for j in range(len(scenario)):
    for i in range(4):
        ax[i, j].grid()
        ax[i, j].set_xlim([0, 2])
        ax[i, j].set_ylim([0, ymax[i]])
        ax[i, j].set_xticks([0, 1, 2])
        ax[i, j].set_xticklabels([])
        ax[i, j].set_axisbelow(True)
        ax[i, j].tick_params(labelsize=12)
plt.savefig('out.pdf')
