#!/usr/bin/env python
import numpy as np
import rankine_vortex as rv
import matplotlib.pyplot as plt

ni = 128  # number of grid points i, j directions
nj = 128
nv = 2   # number of variables, (u, v)
dx = 9000
nr_total = 500
casename = 'perfect_model'
out_cases = ('EnSRF_s1', 'EnSRF_s2', 'EnSRF_s4', 'EnSRF_s8')
colors = ((0.5, 0.5, 0.5), (0.5, 0.5, 1.0), (1.0, 0.5, 0.5), 'c', 'y', (0.5, 1.0, 0.5))
nc = len(out_cases)
p = 1 ##prior(0) or post(1)

loc_rmse_out = np.zeros((nc, nr_total))
wind_rmse_out = np.zeros((nc, nr_total))
state_rmse_out = np.zeros((nc, nr_total))

for nrealize in range(nr_total):
  outdir = '/glade/scratch/mying/rankine/cycle/'+casename+'/{:03d}/'.format(nrealize+1)

  X = np.load(outdir+'truth_state.npy')
  loc = np.load(outdir+'truth_ij.npy')
  wind = np.load(outdir+'truth_wind.npy')

  for c in range(nc):
    Xens = np.load(outdir+out_cases[c]+'_ens.npy')
    nX, nens, nn, nt = Xens.shape
    loc_ens = np.load(outdir+out_cases[c]+'_ij.npy')
    wind_ens = np.load(outdir+out_cases[c]+'_wind.npy')
    loc_rmse = 0
    wind_rmse = 0
    for m in range(nens):
      loc_rmse += np.nanmean((loc[0, 4:nt] - loc_ens[0, m, p, 4:nt])**2 + (loc[1, 4:nt] - loc_ens[1, m, p, 4:nt])**2)
      wind_rmse += np.nanmean((wind[4:nt] - wind_ens[m, p, 4:nt])**2)
    loc_rmse = np.sqrt(loc_rmse/nens)
    wind_rmse = np.sqrt(wind_rmse/nens)
    state_rmse = np.sqrt(np.nanmean(np.load(outdir+out_cases[c]+'_state_err.npy')[p, 4:nt]**2))

    loc_rmse_out[c, nrealize] = loc_rmse
    wind_rmse_out[c, nrealize] = wind_rmse
    state_rmse_out[c, nrealize] = state_rmse

plt.figure(figsize=(5, 10))
ax = plt.subplot(311)
for c in range(nc):
  bx = ax.boxplot(loc_rmse_out[c, :], positions=[c+1], widths=0.5, patch_artist=True, sym='')
  for item in ['boxes', 'whiskers', 'medians', 'caps']:
    plt.setp(bx[item], color='k', linestyle='solid')
  plt.setp(bx['boxes'], facecolor=colors[c])
ax.set_xlim(-1, 6)
ax = plt.subplot(312)
for c in range(nc):
  bx = ax.boxplot(wind_rmse_out[c, :], positions=[c+1], widths=0.5, patch_artist=True, sym='')
  for item in ['boxes', 'whiskers', 'medians', 'caps']:
    plt.setp(bx[item], color='k', linestyle='solid')
  plt.setp(bx['boxes'], facecolor=colors[c])
ax.set_xlim(-1, 6)
ax = plt.subplot(313)
for c in range(nc):
  bx = ax.boxplot(state_rmse_out[c, :], positions=[c+1], widths=0.5, patch_artist=True, sym='')
  for item in ['boxes', 'whiskers', 'medians', 'caps']:
    plt.setp(bx[item], color='k', linestyle='solid')
  plt.setp(bx['boxes'], facecolor=colors[c])
ax.set_xlim(-1, 6)
# ax.grid()
# plt.show()
plt.savefig('1.pdf')
