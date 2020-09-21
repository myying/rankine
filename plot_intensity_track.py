#!/usr/bin/env python
import numpy as np
import rankine_vortex as rv
import matplotlib.pyplot as plt
import sys

ni = 128  # number of grid points i, j directions
nj = 128
nv = 2   # number of variables, (u, v)
dx = 9000

casename = sys.argv[1]
nrealize = int(sys.argv[2])
filter_kind = sys.argv[3] #'EnSRF_s3'
ns = int(sys.argv[4])

outdir = '/glade/scratch/mying/rankine/cycle/'+casename+'/{:03d}/'.format(nrealize)
X = np.load(outdir+'truth_state.npy')
loc = np.load(outdir+'truth_ij.npy')
wind = np.load(outdir+'truth_wind.npy')

Xens = np.load(outdir+filter_kind+'_s{}'.format(ns)+'_ens.npy')
loc_ens = np.load(outdir+filter_kind+'_s{}'.format(ns)+'_ij.npy')
wind_ens = np.load(outdir+filter_kind+'_s{}'.format(ns)+'_wind.npy')
mean_state_err = np.load(outdir+filter_kind+'_s{}'.format(ns)+'_state_err.npy')

nX, nens, nc, nt = Xens.shape
cmap = [plt.cm.jet(x) for x in np.linspace(0, 1,nens)]
ii, jj = np.mgrid[0:ni, 0:nj]
plt.figure(figsize=(10, 6))

####single realization
#track errors
ax = plt.subplot(231)
for m in range(nens):
  ax.plot(loc_ens[0, m, 1, 0:nt], loc_ens[1, m, 1, 0:nt], color=cmap[m][0:3], marker=None)  ##member
ax.plot(loc_ens[0, nens, 0:nt], loc_ens[1, nens, 0:nt], 'g', linewidth=3) ##ens mean
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
#intensity errors
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
ax.plot(mean_state_err[1, 0:nt])
ax.set_ylim(0, 10)
# plt.savefig('{:03d}.png'.format(nrealize), dpi=100)
plt.show()
