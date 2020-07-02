#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import rankine_vortex as rv

filter_kind = ('NoDA', 'EnSRF', 'EnSRF_MSA', 'PF')
colors = ((0.5, 0.5, 0.5), (0.5, 0.5, 1.0), (1.0, 0.5, 0.5), (0.5, 1.0, 0.5))
# params = (20, 40, 80, 160) # ensemble size
# params = (1, 2, 3, 4, 5, 6, 7, 8) # Csprd
params = (2, 5, 7, 10, 15, 20) #obsR
nens = 40
Csprd = 3
obsR = 5

plt.switch_backend('Agg')
plt.figure(figsize=(8, 4))
ax = plt.subplot(1, 1, 1)
for j in range(len(params)):
  for k in range(len(filter_kind)):
    # rmse = np.load('out/diag/mean_error/{}_Csprd{}_N{}_obsR{}.npy'.format(filter_kind[k], Csprd, params[j], obsR))
    # rmse = np.load('out/diag/mean_error/{}_Csprd{}_N{}_obsR{}.npy'.format(filter_kind[k], params[j], nens, obsR))
    rmse = np.load('out/diag/mean_error/{}_Csprd{}_N{}_obsR{}.npy'.format(filter_kind[k], Csprd, nens, params[j]))
    bx = ax.boxplot(rmse, positions=[j*6+k], widths=0.5, patch_artist=True, sym='')
    for item in ['boxes', 'whiskers', 'medians', 'caps']:
      plt.setp(bx[item], color='k', linestyle='solid')
    plt.setp(bx['boxes'], facecolor=colors[k])
ax.set_xlim(-1, len(params)*6-2)
ax.set_xticks(np.arange(len(params))*6+1.5)
ax.set_xticklabels(params)
ax.grid()
plt.savefig('out/1.pdf')
