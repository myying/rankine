#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import rankine_vortex as rv
import graphics as g
import sys

filter_kind = ('none', 'EnSRF', 'EnSRF_MSA', 'PF')
label = ('(a) Prior', '(b) EnSRF', '(c) EnSRF+MSA', '(d) PF')
localize_cutoff = 0

ni = 128  # number of grid points i, j directions
nj = 128
nv = 2   # number of variables, (u, v)
nens = int(sys.argv[1]) # ensemble size
Csprd = int(sys.argv[2])

### Rankine Vortex definition, truth
Rmw = 5    # radius of maximum wind
Vmax = 50   # maximum wind speed
Vout = 0    # wind speed outside of vortex
iStorm = 63 # location of vortex in i, j
jStorm = 63

##truth
iX, jX = rv.make_coords(ni, nj)
Xt = rv.make_state(ni, nj, nv, iStorm, jStorm, Rmw, Vmax, Vout)

plt.switch_backend('Agg')
plt.figure(figsize=(10, 10))
cmap = [plt.cm.jet(x) for x in np.linspace(0, 1,nens)]

i = int(sys.argv[3])
for k in range(len(filter_kind)):
  Xa = np.load('out/{}_Csprd{}_N{}.npy'.format(filter_kind[k], Csprd, nens))[i, :, :]

  ax = plt.subplot(2, 2, k+1)
  for n in range(nens):
    g.plot_contour(ax,ni,nj, Xa[n, :], [cmap[n][0:3]], 1)
  g.plot_contour(ax, ni, nj, Xt, 'black', 3)
  g.set_axis(ax,ni,nj)
  ax.set_title(label[k], fontsize=22, loc='left')
  ax.tick_params(labelsize=15)

plt.savefig('out/{:03d}.png'.format(i), dpi=100)
# plt.show()

