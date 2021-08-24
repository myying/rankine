#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt

filter_kind = ('NoDA_s1', 'EnSRF_s1', 'EnSRF_s2', 'EnSRF_s3', 'EnSRF_s4', 'EnSRF_s5', 'EnSRF_s6', 'EnSRF_s7', 'PF_s1')
colors = ((0.5, 0.5, 0.5), (0.5, 0.5, 1.0), (1.0, 0.5, 0.5), 'c', 'y', (0.5, 1.0, 0.5), 'r', 'k', (.7, .7, .7))
# params = (20, 40, 80) # ensemble size
# params = (1, 2, 3, 4, 5, 6, 7, 8) # Csprd
params = (5,)
# params = (1, 2, 3, 4, 5, 6, 8, 10, 15, 20) #obsR
nens = 20
Csprd = 3
obsR = 5

plt.figure(figsize=(8, 4))
ax = plt.subplot(1, 1, 1)
for j in range(len(params)):
    for k in range(len(filter_kind)):
        rmse = np.load('output/single_obs/{}_N{}_l{}_r{}.npy'.format(filter_kind[k], nens, Csprd, params[j]))[0:900, nens, 0]
        bx = ax.boxplot(rmse, positions=[j*6+k], widths=0.2, showfliers=False, patch_artist=True)
        for item in ['boxes', 'whiskers', 'medians', 'caps']:
            plt.setp(bx[item], color='k', linestyle='solid')
        plt.setp(bx['boxes'], facecolor=None)
ax.set_xlim(-1, len(params)*8)
ax.set_xticks(np.arange(len(params))*8+1.5)
ax.set_xticklabels(params)
ax.grid()
plt.savefig('out.pdf')
