#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from config import *

cases = ('NoDA', 'EnSRF_s1', 'EnSRF_s2', 'EnSRF_s2_mso', 'EnSRF_s3', 'EnSRF_s3_mso', 'EnSRF_s4', 'EnSRF_s4_mso')
nens = 20
Lsprd = (1, 5)
expname = ('full_network/type1', 'full_network/type2')
scenario = ('Vsprd0/Rsprd0.0/phase1.0', 'Vsprd0/Rsprd0.0/phase1.0')
nreal = 100

fig, ax = plt.subplots(4, 2, figsize=(7, 10))
for j in range(len(scenario)):
    for l in range(len(Lsprd)):
        for c in range(len(cases)):
            rmse = np.zeros((nreal, 4))
            r1 = 0
            for r in range(nreal):
                rmse[r1, 0] = np.mean(np.load('output/'+expname[j]+'/{:04d}/Lsprd{}/{}/N{}/{}.npy'.format(r+1, Lsprd[l], scenario[j], nens, cases[c]))[nens, 0])
                rmse[r1, 1] = np.mean(np.load('output/'+expname[j]+'/{:04d}/Lsprd{}/{}/N{}/{}.npy'.format(r+1, Lsprd[l], scenario[j], nens, cases[c]))[0:nens, 1])*9
                rmse[r1, 2] = np.mean(np.load('output/'+expname[j]+'/{:04d}/Lsprd{}/{}/N{}/{}.npy'.format(r+1, Lsprd[l], scenario[j], nens, cases[c]))[0:nens, 2])
                rmse[r1, 3] = np.mean(np.load('output/'+expname[j]+'/{:04d}/Lsprd{}/{}/N{}/{}.npy'.format(r+1, Lsprd[l], scenario[j], nens, cases[c]))[0:nens, 3])*9
                r1 += 1
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
ymax = (1.8, 50, 8, 10)
for j in range(len(scenario)):
    for i in range(4):
        ax[i, j].grid()
        ax[i, j].set_xlim([0, len(Lsprd)])
        ax[i, j].set_ylim([0, ymax[i]])
        ax[i, j].set_xticks([0, 1, 2])
        ax[i, j].set_xticklabels([])
        ax[i, j].set_axisbelow(True)
        ax[i, j].tick_params(labelsize=12)
plt.savefig('out.pdf')
