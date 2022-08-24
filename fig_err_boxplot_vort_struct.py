#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from config import *

cases = ('NoDA', 'EnSRF_s1', 'EnSRF_s2', 'EnSRF_s3', 'EnSRF_s4', 'EnSRF_s5', 'EnSRF_s6')
nens = 20
Lsprd = 3
Vsprd = (2, 4, 10)
Rsprd = 0.0 #(0.5, 2.0, 4.0)
expname = 'full_network/type1'
scenario = 'phase1.0'
nreal = 100

fig, ax = plt.subplots(4, 1, figsize=(5, 12))
for l in range(len(Vsprd)):
    for c in range(len(cases)):
        rmse = np.zeros((nreal, 4))
        r1 = 0
        for r in range(nreal):
            rmse[r1, 0] = np.mean(np.load('output/'+expname+'/{:04d}/Lsprd{}/Vsprd{}/Rsprd{}/{}/N{}/{}.npy'.format(r+1, Lsprd, Vsprd[l], Rsprd, scenario, nens, cases[c]))[nens, 0])
            rmse[r1, 1] = np.mean(np.load('output/'+expname+'/{:04d}/Lsprd{}/Vsprd{}/Rsprd{}/{}/N{}/{}.npy'.format(r+1, Lsprd, Vsprd[l], Rsprd, scenario, nens, cases[c]))[0:nens, 1])*9
            rmse[r1, 2] = np.mean(np.load('output/'+expname+'/{:04d}/Lsprd{}/Vsprd{}/Rsprd{}/{}/N{}/{}.npy'.format(r+1, Lsprd, Vsprd[l], Rsprd, scenario, nens, cases[c]))[0:nens, 2])
            rmse[r1, 3] = np.mean(np.load('output/'+expname+'/{:04d}/Lsprd{}/Vsprd{}/Rsprd{}/{}/N{}/{}.npy'.format(r+1, Lsprd, Vsprd[l], Rsprd, scenario, nens, cases[c]))[0:nens, 3])*9
            r1 += 1
        for i in range(4):
            x = l + c*0.12 + 0.1
            q3, q1 = np.percentile(rmse[0:r1, i], [75, 25])
            median = np.median(rmse[0:r1, i])
            fc = [1, 1, 1]
            if c==0:
                fc = [.7, .7, .7]
            if c in (2, 3, 4, 5, 6, 7):
                fc = [0, .7, .85]
            ax[i].add_patch(Polygon([(x-0.04,q1), (x-0.04,q3), (x+0.04,q3), (x+0.04,q1)], facecolor=fc, ec='black'))
            ax[i].plot(x, median, marker='.', color='black')
ymax = (1.8, 45, 8, 20)
for i in range(4):
    ax[i].grid()
    ax[i].set_xlim([0, len(Vsprd)])
    ax[i].set_ylim([0, ymax[i]])
    ax[i].set_xticklabels([])
    ax[i].set_axisbelow(True)
    ax[i].tick_params(labelsize=12)
plt.savefig('out.pdf')
