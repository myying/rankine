#!/usr/bin/env python
import numpy as np
import rankine_vortex as rv
import matplotlib.pyplot as plt

outdir = '/Users/mying/work/rankine/cycle/'
ni = 128  # number of grid points i, j directions
nj = 128
nv = 2   # number of variables, (u, v)
dx = 9000
nt = 5
cmap = [plt.cm.jet(x) for x in np.linspace(0, 1,nt)]

X = np.load(outdir+'truth_state.npy')

obs = np.load(outdir+'obs.npy')
iObs = np.load(outdir+'obs_i.npy')
jObs = np.load(outdir+'obs_j.npy')
nt1, nobs1 = obs.shape
nobs = int(nobs1/2)

ii, jj = np.mgrid[0:ni, 0:nj]
u, v = rv.X2uv(ni, nj, X[:, -1])
zeta = rv.uv2zeta(u, v, dx)

plt.figure(figsize=(10, 8))
ax = plt.subplot(111)
for n in range(nt):
  ax.scatter(iObs[n, ::2], jObs[n, ::2], s=50, color=[cmap[n][0:3]])
#for p in range(nobs):

ax.set_xlim(0, ni)
ax.set_ylim(0, nj)

plt.savefig('1.pdf')
