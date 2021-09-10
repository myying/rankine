#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt

cases = ('NoDA', 'EnSRF_s1', 'EnSRF_s2', 'EnSRF_s3', 'EnSRF_s4', 'EnSRF_s5', 'EnSRF_s6', 'EnSRF_s7', 'PF')
#cases = ('NoDA', 'EnSRF_s1', 'PF')
nens = 200
Csprd = (1, 2, 3, 4, 5)
color = ('r', 'g', 'b', 'c', 'y')
nreal = 200

plt.figure(figsize=(8, 4))
ax = plt.subplot(1, 1, 1)
for l in range(len(Csprd)):
    out = np.zeros(len(cases))
    for c in range(len(cases)):
        rmse = np.zeros(nreal)
        for r in range(nreal):
            rmse[r] = np.mean(np.load('output/single_obs/L{}/{:04d}/{}_err.npy'.format(Csprd[l], r+1, cases[c]))[0:nens, 2])
        bx = ax.boxplot(rmse, positions=[c+l*0], widths=0.15, showfliers=False, patch_artist=True)
        for item in ['boxes', 'whiskers', 'medians', 'caps']:
            plt.setp(bx[item], color=color[l], linestyle='solid')
        plt.setp(bx['boxes'], facecolor=color[l])
        out[c] = np.mean(rmse)
    ax.plot(out, color[l])
ax.set_xlim(-1, 9)
# ax.set_ylim(0, 2)
ax.grid()
plt.savefig('out.pdf')
