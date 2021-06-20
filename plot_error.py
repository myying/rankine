#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import rankine_vortex as rv

filter_kind = ('NoDA_s1', 'EnSRF_s1', 'EnSRF_s2', 'EnSRF_s3', 'EnSRF_s4', 'EnSRF_s7', 'EnSRF_s6', 'PF_s1')
# filter_kind = ('NoDA_s1', 'EnSRF_s1', 'PF_s1')
colors = ((0.5, 0.5, 0.5), (0.5, 0.5, 1.0), (1.0, 0.5, 0.5), 'c', 'y', (0.5, 1.0, 0.5), 'r', (.7, .7, .7))
# params = (20, 40, 80) # ensemble size
# params = (1, 2, 3, 4, 5, 6, 7, 8) # Csprd
params = (5,)
# params = (1, 2, 3, 4, 5, 6, 8, 10, 15, 20) #obsR
nens = 40
Csprd = 5
obsR = 5
obserr = 5

plt.figure(figsize=(8, 4))
ax = plt.subplot(1, 1, 1)
for j in range(len(params)):
    for k in range(len(filter_kind)):
        rmse = np.load('/storage/windows10/scratch/rankine/single/{}_N{}_C{}_R{}_err{}.npy'.format(filter_kind[k], nens, Csprd, params[j], obserr))[0:100, nens]
        bx = ax.boxplot(rmse, positions=[j*6+k], widths=0.5, patch_artist=True, sym='')
        for item in ['boxes', 'whiskers', 'medians', 'caps']:
            plt.setp(bx[item], color='k', linestyle='solid')
        plt.setp(bx['boxes'], facecolor=colors[k])
ax.set_xlim(-1, len(params)*8)
ax.set_xticks(np.arange(len(params))*8+1.5)
ax.set_xticklabels(params)
ax.grid()
# plt.savefig('0.pdf')
plt.show()
