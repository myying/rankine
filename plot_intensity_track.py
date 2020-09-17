#!/usr/bin/env python
import numpy as np
import rankine_vortex as rv
import matplotlib.pyplot as plt
import sys

outdir = '/glade/scratch/mying/rankine/cycle/'
ni = 128  # number of grid points i, j directions
nj = 128
nv = 2   # number of variables, (u, v)
dx = 9000

casename = sys.argv[1] #'EnSRF_s3'
X = np.load(outdir+'truth_state.npy')
loc = np.load(outdir+'truth_ij.npy')
wind = np.load(outdir+'truth_wind.npy')
Xens = np.load(outdir+casename+'_ens.npy')
loc_ens = np.load(outdir+casename+'_ij.npy')
wind_ens = np.load(outdir+casename+'_wind.npy')
nX, nens, nc, nt = Xens.shape
cmap = [plt.cm.jet(x) for x in np.linspace(0, 1,nens)]

ii, jj = np.mgrid[0:ni, 0:nj]
u, v = rv.X2uv(ni, nj, X[:, -1])
zeta = rv.uv2zeta(u, v, dx)

plt.figure(figsize=(10, 6))

##track errors
ax = plt.subplot(231)
# c = ax.contourf(ii, jj, zeta, np.arange(-3, 3, 0.1)*1e-3, cmap='bwr')
# c = ax.contourf(ii, jj, u, np.arange(-70, 70, 5), cmap='bwr')
for m in range(nens):
  ax.plot(loc_ens[0, m, 1, 0:nt], loc_ens[1, m, 1, 0:nt], color=cmap[m][0:3], marker=None)  ##member
# ax.plot(loc[0, nens, 0:nt], loc_ens[1, nens, 0:nt], 'g', linewidth=3) ##ens mean
ax.plot(loc[0, 0:nt], loc[1, 0:nt], 'k', linewidth=3)  ##true
ax.set_xlim(0, ni)
ax.set_ylim(0, nj)

ax = plt.subplot(234)
loc_rmse = 0
for m in range(nens):
  loc_rmse += (loc[0, 0:nt] - loc_ens[0, m, 1, 0:nt])**2 + (loc[1, 0:nt] - loc_ens[1, m, 1, 0:nt])**2
loc_rmse = np.sqrt(loc_rmse/nens)
ax.plot(loc_rmse)
# loc_m_rmse = np.sqrt((loc[0, 0:nt] - loc_ens[0, nens, 1, 0:nt])**2 + (loc[1, 0:nt] - loc_ens[1, nens, 1, 0:nt])**2)
# ax.plot(loc_m_rmse)
ax.set_ylim(0, 20)

##intensity errors
ax = plt.subplot(232)
for m in range(nens):
  ax.plot(wind_ens[m, 1, 0:nt], color=cmap[m][0:3], marker=None)
ax.plot(wind[0:nt], 'k', linewidth=3)
ax.set_ylim(10, 70)

ax = plt.subplot(235)
wind_rmse = 0
for m in range(nens):
  wind_rmse += (wind[0:nt] - wind_ens[m, 1, 0:nt])**2
wind_rmse = np.sqrt(wind_rmse/nens)
ax.plot(wind_rmse)
ax.set_ylim(0, 20)

##physics space rmse
ax = plt.subplot(236)
rmse = np.zeros(nt)
for t in range(nt):
  um, vm = rv.X2uv(ni, nj, np.mean(Xens[:, 0:nens, 1, t], axis=1))
  ut, vt = rv.X2uv(ni, nj, X[:, t])
  sq_err = (um - ut)**2 + (vm - vt)**2
  loc_i = int(loc[0, t])
  loc_j = int(loc[1, t])
  buff = 10
  rmse[t] = np.sqrt(np.mean(sq_err[loc_i-buff:loc_i+buff, loc_j-buff:loc_j+buff]))
ax.plot(rmse)
ax.set_ylim(0, 6)

# plt.savefig('1.pdf')
plt.show()
