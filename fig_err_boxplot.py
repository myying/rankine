#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from config import *

cases = ('NoDA', 'EnSRF_s1', 'EnSRF_s2', 'EnSRF_s3', 'EnSRF_s4', 'EnSRF_s5', 'EnSRF_s6', 'EnSRF_s7', 'PF')
nens = (20, 20, 20, 20, 20, 20, 20, 20, 500)
Lsprd = (1, 2, 3, 4, 5)
Lbias = 0
expname = ('single_wind_obs', 'position_obs')
nreal = (100, 100)

fig, ax = plt.subplots(4, 2, figsize=(12, 12))
for j in range(2):
    for l in range(len(Lsprd)):
        for c in range(len(cases)):
            rmse = np.zeros((nreal[j], 4))
            r1 = 0
            for r in range(nreal[j]):
                rmse[r1, 0] = np.mean(np.load('output/'+expname[j]+'/{:04d}/Lbias{}/Lsprd{}/N{}/{}.npy'.format(r+1, Lbias, Lsprd[l], nens[c], cases[c]))[nens[c], 0])
                rmse[r1, 1] = np.mean(np.load('output/'+expname[j]+'/{:04d}/Lbias{}/Lsprd{}/N{}/{}.npy'.format(r+1, Lbias, Lsprd[l], nens[c], cases[c]))[0:nens[c], 1])*9
                rmse[r1, 2] = np.mean(np.load('output/'+expname[j]+'/{:04d}/Lbias{}/Lsprd{}/N{}/{}.npy'.format(r+1, Lbias, Lsprd[l], nens[c], cases[c]))[0:nens[c], 2])
                rmse[r1, 3] = np.mean(np.load('output/'+expname[j]+'/{:04d}/Lbias{}/Lsprd{}/N{}/{}.npy'.format(r+1, Lbias, Lsprd[l], nens[c], cases[c]))[0:nens[c], 3])*9
                r1 += 1
            for i in range(4):
                x = l + c*0.1 + 0.1
                q3, q1 = np.percentile(rmse[0:r1, i], [75, 25])
                median = np.median(rmse[0:r1, i])
                if c==0:  ##NoDA
                    fc = [.7, .7, .7]
                if c==len(cases)-1:  ##PF
                    fc = [.5, .5, .5]
                if c==1:  ##EnSRF
                    fc = [.5, .9, .5]
                if c>1 and c<len(cases)-1:  ##EnSRF_MSA Ns>1
                    fc = [0, .7, .85]
                ax[i, j].add_patch(Polygon([(x-0.04,q1), (x-0.04,q3), (x+0.04,q3), (x+0.04,q1)], facecolor=fc, ec=None))
                ax[i, j].plot(x, median, marker='.', color='black')
ymax = (1.8, 45, 6, 18)
for j in range(2):
    for i in range(4):
        ax[i, j].grid()
        ax[i, j].set_xlim([0, len(Lsprd)])
        ax[i, j].set_ylim([0, ymax[i]])
        ax[i, j].set_xticklabels([])
        ax[i, j].set_axisbelow(True)
        ax[i, j].tick_params(labelsize=12)
plt.savefig('out.pdf')
