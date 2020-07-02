#!/usr/bin/env python
import numpy as np
import rankine_vortex as rv
import matplotlib.pyplot as plt

np.random.seed(1)

ni = 128  # number of grid points i, j directions
nj = 128
nv = 2   # number of variables, (u, v)
dx = 9000
dt = 60

### Rankine Vortex definition, truth
Rmw = 5    # radius of maximum wind
Vmax = 50   # maximum wind speed
Vout = 0    # wind speed outside of vortex
iStorm = 63 # location of vortex in i, j
jStorm = 63

##initial state
iX, jX = rv.make_coords(ni, nj)
X = rv.make_state1(ni, nj, nv, dx, iStorm, jStorm, Rmw)

for n in range(12):
  u, v = rv.X2uv(ni, nj, X)
  zeta = rv.uv2zeta(u, v, dx)

  plt.figure(figsize=(13, 5))
  ii, jj = np.mgrid[0:ni, 0:nj]
  ax = plt.subplot(121)
  c = ax.contourf(ii, jj, zeta, np.arange(-10, 10, 1)*1e-4, cmap='bwr')
  plt.colorbar(c)
  ax.set_aspect('equal', 'box')
  ax.set_xlim(0, ni)
  ax.set_ylim(0, nj)
  ax = plt.subplot(122)
  qv = ax.quiver(ii[::4, ::4], jj[::4, ::4], u[::4, ::4], v[::4, ::4], scale=500, headwidth=3, headlength=5, headaxislength=5, linewidths=0.5)
  ax.set_aspect('equal', 'box')
  ax.set_xlim(0, ni)
  ax.set_ylim(0, nj)
  plt.savefig('out/{:03d}.png'.format(n), dpi=100)
  plt.close()

  X = rv.advance_time(ni, nj, X, dx, 60, dt)

