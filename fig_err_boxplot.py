#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from config import *

cases = ('NoDA', 'EnSRF_s1', 'EnSRF_s2', 'EnSRF_s3', 'EnSRF_s4', 'EnSRF_s5', 'EnSRF_s6', 'EnSRF_s7', 'PF')
nens = 20
Csprd = (1, 2, 3, 4, 5)
expname = ('single_wind_obs', 'position_obs')
nreal = (1000, 1000)

fig, ax = plt.subplots(4, 2, figsize=(12, 12))
for j in range(2):
    for l in range(len(Csprd)):
        for c in range(len(cases)):
            rmse = np.zeros((nreal[j], 4))
            r1 = 0
            for r in range(nreal[j]):
                #     expname = 'single_wind_obs'
                #     Yloc = np.load('output/'+expname+'/{:04d}/Yloc.npy'.format(r+1))
                #     obsR = np.sqrt((Yloc[0, 0]-0.5*ni)**2 + (Yloc[1, 0]-0.5*nj)**2)
                #     if obsR<0 or obsR>10: ##check obs radius, only for
                #         continue
                rmse[r1, 0] = np.mean(np.load('output/'+expname[j]+'/{:04d}/Lbias0/Lsprd{}/N{}/{}.npy'.format(r+1, Csprd[l], nens, cases[c]))[nens, 0])
                rmse[r1, 1] = np.mean(np.load('output/'+expname[j]+'/{:04d}/Lbias0/Lsprd{}/N{}/{}.npy'.format(r+1, Csprd[l], nens, cases[c]))[0:nens, 1])*9
                rmse[r1, 2] = np.mean(np.load('output/'+expname[j]+'/{:04d}/Lbias0/Lsprd{}/N{}/{}.npy'.format(r+1, Csprd[l], nens, cases[c]))[0:nens, 2])
                rmse[r1, 3] = np.mean(np.load('output/'+expname[j]+'/{:04d}/Lbias0/Lsprd{}/N{}/{}.npy'.format(r+1, Csprd[l], nens, cases[c]))[0:nens, 3])*9
                r1 += 1
            for i in range(4):
                x = l + c*0.1 + 0.1
                q3, q1 = np.percentile(rmse[0:r1, i], [75, 25])
                median = np.median(rmse[0:r1, i])
                if c==0 or c==len(cases)-1:
                    fc = [.7, .7, .7]
                else:
                    fc = [1, 1, 1]
                ax[i, j].add_patch(Polygon([(x-0.04,q1), (x-0.04,q3), (x+0.04,q3), (x+0.04,q1)], facecolor=fc, ec='black'))
                ax[i, j].plot(x, median, marker='.', color='black')
ymax = (1.8, 45, 6, 18)
for j in range(2):
    for i in range(4):
        ax[i, j].grid()
        ax[i, j].set_xlim([0, len(Csprd)])
        ax[i, j].set_ylim([0, ymax[i]])
        ax[i, j].set_xticklabels([])
        ax[i, j].set_axisbelow(True)
        ax[i, j].tick_params(labelsize=12)
plt.savefig('out.pdf')
