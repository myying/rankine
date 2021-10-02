#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from config import *

cases = ('NoDA_s1_1', 'EnSRF_s1_1', 'EnSRF_s2_1', 'EnSRF_s2_2', 'EnSRF_s3_1', 'EnSRF_s3_3', 'EnSRF_s4_1', 'EnSRF_s4_4')
nens = (30, 30, 28, 28, 25, 25, 22, 22)
expname = ('cycling/perfect_model/type2', 'cycling/perfect_model/type2', 'cycling/imperfect_model/type2')
scenario = ('Lsprd3/phase1.0', 'Lsprd3/phase0.0', 'Lsprd3/phase1.0')
nreal = (100, 100, 100)

fig, ax = plt.subplots(4, 3, figsize=(12, 12))
k = 0  ##0=prior, 1=posterior
for j in range(3):
    for l in (0, 1):
        for c in range(len(cases)):
            rmse = np.zeros((nreal[j], 4))
            r1 = 0
            for r in range(nreal[j]):
                rmse[r1, 0] = np.mean(np.load('output/'+expname[j]+'/{:04d}/{}/{}.npy'.format(r+1, scenario[j], cases[c]))[-1, 0, l, ::3])
                rmse[r1, 1] = np.mean(np.load('output/'+expname[j]+'/{:04d}/{}/{}.npy'.format(r+1, scenario[j], cases[c]))[0:nens[c], 1, l, ::3])*9
                rmse[r1, 2] = np.mean(np.load('output/'+expname[j]+'/{:04d}/{}/{}.npy'.format(r+1, scenario[j], cases[c]))[0:nens[c], 2, l, ::3])
                rmse[r1, 3] = np.mean(np.load('output/'+expname[j]+'/{:04d}/{}/{}.npy'.format(r+1, scenario[j], cases[c]))[0:nens[c], 3, l, ::3])*9
                r1 += 1
            for i in range(4):
                x = l + c*0.1 + 0.1
                q3, q1 = np.percentile(rmse[0:r1, i], [75, 25])
                median = np.median(rmse[0:r1, i])
                fc = [1, 1, 1]
                if c==0:
                    fc = [.7, .7, .7]
                if c in (2, 4, 6):
                    fc = [.9, .8, .6]
                if c in (3, 5, 7):
                    fc = [.8, .4, .4]
                ax[i, j].add_patch(Polygon([(x-0.04,q1), (x-0.04,q3), (x+0.04,q3), (x+0.04,q1)], facecolor=fc, ec='black'))
                ax[i, j].plot(x, median, marker='.', color='black')
ymax = (3.5, 60, 10, 10)
for j in range(3):
    for i in range(4):
        ax[i, j].grid()
        ax[i, j].set_xlim([0, 2])
        ax[i, j].set_ylim([0, ymax[i]])
        ax[i, j].set_xticks([0, 1, 2])
        ax[i, j].set_xticklabels([])
        ax[i, j].set_axisbelow(True)
        ax[i, j].tick_params(labelsize=12)
plt.savefig('out.pdf')
