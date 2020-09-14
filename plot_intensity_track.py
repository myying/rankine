#!/usr/bin/env python
import numpy as np
import rankine_vortex as rv
import matplotlib.pyplot as plt
import sys

outdir = '/Users/mying/work/rankine/cycle/'
ni = 128  # number of grid points i, j directions
nj = 128
nv = 2   # number of variables, (u, v)
dx = 9000
nt = 9
nens = 20
cmap = [plt.cm.jet(x) for x in np.linspace(0, 1,nens)]

casename = sys.argv[1] #'EnSRF_s3'
X = np.load(outdir+'truth_state.npy')
loc = np.load(outdir+'truth_ij.npy')
wind = np.load(outdir+'truth_wind.npy')
Xens = np.load(outdir+casename+'_ens.npy')
loc_ens = np.load(outdir+casename+'_ij.npy')
wind_ens = np.load(outdir+casename+'_wind.npy')

ii, jj = np.mgrid[0:ni, 0:nj]
u, v = rv.X2uv(ni, nj, X[:, -1])
zeta = rv.uv2zeta(u, v, dx)

plt.figure(figsize=(10, 10))
ax = plt.subplot(221)
# c = ax.contourf(ii, jj, zeta, np.arange(-3, 3, 0.1)*1e-3, cmap='bwr')
# c = ax.contourf(ii, jj, u, np.arange(-70, 70, 5), cmap='bwr')
for m in range(nens):
  ax.plot(loc_ens[0, m, 1, 0:nt], loc_ens[1, m, 1, 0:nt], color=cmap[m][0:3], marker=None)
ax.plot(loc[0, :], loc[1, :], 'k', linewidth=3)
ax.set_xlim(0, ni)
ax.set_ylim(0, nj)

ax = plt.subplot(223)
loc_rmse = 0
for m in range(nens):
  loc_rmse += (loc[0, 0:nt] - loc_ens[0, m, 1, 0:nt])**2 + (loc[1, 0:nt] - loc_ens[1, m, 1, 0:nt])**2
loc_rmse = np.sqrt(loc_rmse/nens)
ax.plot(loc_rmse)
ax.set_ylim(0, 20)

ax = plt.subplot(222)
for m in range(nens):
  ax.plot(wind_ens[m, 1, 0:nt], color=cmap[m][0:3], marker=None)
ax.plot(wind, 'k', linewidth=3)

ax = plt.subplot(224)
wind_rmse = 0
for m in range(nens):
  wind_rmse += (wind[0:nt] - wind_ens[m, 1, 0:nt])**2
wind_rmse = np.sqrt(wind_rmse/nens)
ax.plot(wind_rmse)
ax.set_ylim(0, 20)

# plt.savefig('1.pdf')
plt.show()
