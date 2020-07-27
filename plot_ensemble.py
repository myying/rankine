#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import rankine_vortex as rv
import graphics as g
import sys

outdir = '/Users/mying/work/rankine/cycle//'
filter_kind = ('EnSRF_s1', 'EnSRF_s4', 'EnSRF_s8')

ni = 128  # number of grid points i, j directions
nj = 128
nv = 2   # number of variables, (u, v)
nens = 20 #int(sys.argv[1]) # ensemble size
t = int(sys.argv[1])
m = int(sys.argv[2])

### Rankine Vortex definition, truth
Rmw = 5    # radius of maximum wind
Vmax = 50   # maximum wind speed
Vout = 0    # wind speed outside of vortex
iStorm = 63 # location of vortex in i, j
jStorm = 63

##truth
Xt = np.load(outdir+'truth_state.npy')[:, t]

plt.switch_backend('Agg')
plt.figure(figsize=(10, 10))
cmap = [plt.cm.jet(x) for x in np.linspace(0, 1,nens)]

ax = plt.subplot(221)
u, v = rv.X2uv(ni, nj, Xt)
ii, jj = np.mgrid[0:ni, 0:nj]
ax.contourf(ii, jj, u, np.arange(-50, 51, 5), cmap='bwr')
g.plot_contour(ax, ni, nj, Xt, 'black', 3)
g.set_axis(ax, ni, nj)
ax.set_title('Truth', fontsize=20)
ax.tick_params(labelsize=15)

for k in range(len(filter_kind)):
  Xa = np.load(outdir+filter_kind[k]+'_ens.npy')[:, :, t]
  ax = plt.subplot(2, 2, k+2)
  u, v = rv.X2uv(ni, nj, Xa[:, m])
  ax.contourf(ii, jj, u, np.arange(-50, 51, 5), cmap='bwr')
  g.plot_contour(ax, ni, nj, Xt, 'black', 3)
  g.set_axis(ax,ni,nj)
  ax.set_title(filter_kind[k], fontsize=20)
  ax.tick_params(labelsize=15)

plt.savefig('{:03d}.png'.format(m+1), dpi=100)

