#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import rankine_vortex as rv
import sys

outdir = '/glade/scratch/mying/rankine/cycle/'
casename = sys.argv[1] #'EnSRF_s1'

ni = 128  # number of grid points i, j directions
nj = 128
dx = 9000
nv = 2   # number of variables, (u, v)
nens = 20 #int(sys.argv[1]) # ensemble size
t = int(sys.argv[2]) #1
s = int(sys.argv[3]) #1

### Rankine Vortex definition, truth
Rmw = 5    # radius of maximum wind
Vmax = 50   # maximum wind speed
Vout = 0    # wind speed outside of vortex
iStorm = 63 # location of vortex in i, j
jStorm = 63
wind_highlight = (-15, 15)

# plt.switch_backend('Agg')
plt.figure(figsize=(10, 10))
cmap = [plt.cm.jet(x) for x in np.linspace(0, 1,nens)]
ii, jj = np.mgrid[0:ni, 0:nj]

ax = plt.subplot(221)
X = np.load(outdir+casename+'_ens.npy')[:, :, 1, t]
for m in range(nens):
  u, v = rv.X2uv(ni, nj, X[:, m])
  ax.contour(ii, jj, u, wind_highlight, colors=[cmap[m][0:3]], linewidths=1)
Xt = np.load(outdir+'truth_state.npy')[:, t]
u, v = rv.X2uv(ni, nj, Xt)
zeta = rv.uv2zeta(u, v, dx)
# c = ax.contourf(ii, jj, zeta, np.arange(-3, 3, 0.1)*1e-3, cmap='bwr')
# plt.colorbar(c)
ax.contour(ii, jj, u, wind_highlight, colors='k', linewidths=3)
ax.tick_params(labelsize=15)

ax = plt.subplot(222)
dat = np.load(outdir+casename+'_inflation.npy')[:, :, s, t]
inf_mean = np.reshape(dat[0:ni*nj, 0], (ni, nj))
inf_sd = np.reshape(dat[0:ni*nj, 1], (ni, nj))
c = ax.contourf(ii, jj, inf_mean) #, np.arange(0.9, 1.6, 0.01), cmap='jet')
plt.colorbar(c)

ax = plt.subplot(223)
c = ax.contourf(ii, jj, inf_sd) #, np.arange(0.1, 1.2, 0.01), cmap='jet')
plt.colorbar(c)

ax = plt.subplot(224)
dat = np.load(outdir+casename+'_ens.npy')[0:ni*nj, :, 1, t-1]
dat_mean = np.mean(dat, axis=1)
sprd = np.zeros(ni*nj)
for m in range(nens):
  sprd += (dat[:, m] - dat_mean)**2
sprd = sprd / (nens-1)
sprd = np.sqrt(sprd)
sprd_out = np.reshape(sprd, (ni, nj))
c = ax.contourf(ii, jj, sprd_out)
plt.colorbar(c)

# plt.savefig('{:03d}.png'.format(t+1), dpi=100)
plt.show()


