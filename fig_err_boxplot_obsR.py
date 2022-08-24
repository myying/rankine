#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from config import *

cases = ('NoDA', 'EnSRF_s1', 'EnSRF_s2', 'EnSRF_s3', 'EnSRF_s4', 'EnSRF_s5', 'EnSRF_s6', 'EnSRF_s7', 'PF')
nens = 20
obsR = (0, 2, 4, 6, 8, 10)
Lsprd = (1, 2, 3, 4, 5)
Lbias = 0
expname = 'single_wind_obs'
nreal = 1000

fig, ax = plt.subplots(5, 1, figsize=(6, 9))
for l in range(len(Lsprd)):
    for c in range(len(cases)):
        rmse = np.zeros((nreal, 5))
        rt = np.zeros(5).astype(int)
        for r in range(nreal):
            Yloc = np.load('output/'+expname+'/{:04d}/Yloc.npy'.format(r+1))
            R = np.sqrt((Yloc[0, 0]-0.5*ni)**2 + (Yloc[1, 0]-0.5*nj)**2)
            for i in range(5):
                if R>obsR[i] and R<=obsR[i+1]:
                    rmse[rt[i], i] = np.mean(np.load('output/'+expname+'/{:04d}/Lbias{}/Lsprd{}/N{}/{}.npy'.format(r+1, Lbias, Lsprd[l], nens, cases[c]))[nens, 0])
                    rt[i] += 1
        for i in range(5):
            x = l + c*0.1 + 0.1
            q3, q1 = np.percentile(rmse[0:rt[i], i], [75, 25])
            median = np.median(rmse[0:rt[i], i])
            if c==0:  ##NoDA
                fc = [.7, .7, .7]
                ec = [0, 0, 0]
            if c==len(cases)-1:  ##PF
                fc = [0, .5, .3]
                ec = [0, 0, 0]
            if c==1:  ##EnSRF
                fc = [1, 1, 1]
                ec = [0, 0, 0]
            if c>1 and c<len(cases)-1:  ##EnSRF_MSA Ns>1
                fc = [0, .7, .85]
                ec = None
            ax[4-i].add_patch(Polygon([(x-0.04,q1), (x-0.04,q3), (x+0.04,q3), (x+0.04,q1)], facecolor=fc, ec=ec))
            ax[4-i].plot(x, median, marker='.', color='black')
print(rt)
for i in range(5):
    ax[i].set_xlim([0, len(Lsprd)])
    ax[i].set_ylim([0, 2])
    ax[i].set_xticklabels([])
    ax[i].grid(axis='x')
    ax[i].tick_params(labelsize=12)
plt.tight_layout()
plt.savefig('out.pdf')
