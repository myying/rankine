#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import rankine_vortex as rv
import sys

outdir = '/Users/mying/work/rankine/cycle/'
filter_kind = ('NoDA_s1', 'EnSRF_s1') #, 'EnSRF_s4', 'EnSRF_s8')

ni = 128  # number of grid points i, j directions
nj = 128
dx = 9000
nv = 2   # number of variables, (u, v)
nens = 40 #int(sys.argv[1]) # ensemble size
t = int(sys.argv[1])
# m = int(sys.argv[2])

### Rankine Vortex definition, truth
Rmw = 5    # radius of maximum wind
Vmax = 50   # maximum wind speed
Vout = 0    # wind speed outside of vortex
iStorm = 63 # location of vortex in i, j
jStorm = 63
wind_highlight = (-15, 15)

##truth
Xt = np.load(outdir+'truth_state.npy')[:, t]

# plt.switch_backend('Agg')
plt.figure(figsize=(10, 10))
cmap = [plt.cm.jet(x) for x in np.linspace(0, 1,nens)]

ax = plt.subplot(221)
u, v = rv.X2uv(ni, nj, Xt)
zeta = rv.uv2zeta(u, v, dx)
ii, jj = np.mgrid[0:ni, 0:nj]
# ax.contourf(ii, jj, u, np.arange(-70, 71, 5), cmap='bwr')
ax.contourf(ii, jj, zeta, np.arange(-3, 3, 0.1)*1e-3, cmap='bwr')
ax.contour(ii, jj, u, wind_highlight, colors='k', linewidths=3)
ax.set_title('Truth', fontsize=20)
ax.tick_params(labelsize=15)

for k in range(len(filter_kind)):
  Xa = np.load(outdir+filter_kind[k]+'_ens.npy')[:, :, t]
  ax = plt.subplot(2, 2, k+2)
  for m in range(nens):
    u, v = rv.X2uv(ni, nj, Xa[:, m])
    ax.contour(ii, jj, u, wind_highlight, colors=[cmap[m][0:3]], linewidths=1)
  u, v = rv.X2uv(ni, nj, Xt)
  ax.contour(ii, jj, u, wind_highlight, colors='k', linewidths=3)
  ax.set_title(filter_kind[k], fontsize=20)
  ax.tick_params(labelsize=15)

# plt.savefig('{:03d}.png'.format(t+1), dpi=100)
plt.show()


