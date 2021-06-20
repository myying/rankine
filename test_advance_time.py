#!/usr/bin/env python
import numpy as np
import rankine_vortex as rv
import matplotlib.pyplot as plt


ni = 128    # number of grid points i, j directions
nj = 128
nv = 2     # number of variables, (u, v)
dx = 9000
dt = 300
nt = 12
diss = 5*1e3

### Rankine Vortex definition, truth
Rmw = 5        # radius of maximum wind
Vmax = 50     # maximum wind speed
Vout = 0        # wind speed outside of vortex
iStorm = 83 # location of vortex in i, j
jStorm = 53
Csprd = 3
nens = 20
cmap = [plt.cm.jet(m) for m in np.linspace(0.2, 0.8, nens)]

np.random.seed(0)
X_bkg = rv.make_background_flow(ni, nj, nv, dx, ampl=1e-4)

##initial state
np.random.seed(0)
iX, jX = rv.make_coords(ni, nj)
X = np.zeros((ni*nj*nv, nens, nt+1))
iBias = 0
jBias = 0
Rsprd = 0
Vsprd = 0
iStorm_ens = np.zeros(nens)
jStorm_ens = np.zeros(nens)
for m in range(nens):
    iStorm_ens[m] = iStorm + np.random.normal(0, 1) * Csprd
    jStorm_ens[m] = jStorm + np.random.normal(0, 1) * Csprd
    X[:, m, 0] = rv.make_state(ni, nj, nv, iStorm_ens[m], jStorm_ens[m], Rmw, Vmax, Vout)
    X[:, m, 0] += X_bkg + rv.make_background_flow(ni, nj, nv, dx, ampl=5e-5)

for n in range(nt):
    print(n)
    u, v = rv.X2uv(ni, nj, X[:, :, n])
    zeta = rv.uv2zeta(u, v, dx)
    plt.figure(figsize=(13, 5))
    ii, jj = np.mgrid[0:ni, 0:nj]
    ax = plt.subplot(121)
    c = ax.contourf(ii, jj, zeta[:, :, 0], np.arange(-30, 30, 1)*1e-4, cmap='bwr')
    plt.colorbar(c)
    for m in range(nens):
        ax.contour(ii, jj, zeta[:, :, m], (3e-4,), colors=[cmap[m][0:3]], linewidths=2)
    ax.set_aspect('equal', 'box')
    ax.set_xlim(0, ni)
    ax.set_ylim(0, nj)
    ax = plt.subplot(122)
    qv = ax.quiver(ii[::4, ::4], jj[::4, ::4], u[::4, ::4, 0], v[::4, ::4, 0], scale=500, headwidth=3, headlength=5, headaxislength=5, linewidths=0.5)
    ax.set_aspect('equal', 'box')
    ax.set_xlim(0, ni)
    ax.set_ylim(0, nj)
    plt.savefig('{:03d}.png'.format(n), dpi=100)
    plt.close()

    X[:, :, n+1] = rv.advance_time(ni, nj, X[:, :, n], dx, 12, dt, diss)

np.save('ens_NoDAs.npy', X)
