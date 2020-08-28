#!/usr/bin/env python
import numpy as np
import rankine_vortex as rv
import matplotlib.pyplot as plt
import data_assimilation as DA

ni = 128  # number of grid points i, j directions
nj = 128
nv = 2   # number of variables, (u, v)
nens = 80 # ensemble size

### Rankine Vortex definition, truth
Rmw = 5    # radius of maximum wind
Vmax = 50   # maximum wind speed
Vout = 0    # wind speed outside of vortex
iStorm = 63 # location of vortex in i, j
jStorm = 63

##truth
iX, jX = rv.make_coords(ni, nj)
Xt = rv.make_state(ni, nj, nv, iStorm, jStorm, Rmw, Vmax, Vout)

np.random.seed(9)

##Prior ensemble
Csprd = 3.0
iBias = 0
jBias = 0
Rsprd = 0
Vsprd = 0
Xb = np.zeros((nens, ni*nj*nv))
iStorm_ens = np.zeros(nens)
jStorm_ens = np.zeros(nens)
for n in range(nens):
  iStorm_ens[n] = iStorm + iBias + np.random.normal(0, 1) * Csprd
  jStorm_ens[n] = jStorm + jBias + np.random.normal(0, 1) * Csprd
  Rmw_n = Rmw + np.random.normal(0, 1) * Rsprd
  Vmax_n = Vmax + np.random.normal(0, 1) * Vsprd
  Vout_n = Vout + np.random.normal(0, 1) * 0.0
  Xb[n, :] = rv.make_state(ni, nj, nv, iStorm_ens[n], jStorm_ens[n], Rmw_n, Vmax_n, Vout_n)

###Observations
iObs = np.array([66.5, 66.5])
jObs = np.array([66.5, 66.5])
vObs = np.array([0, 1])
nobs = iObs.size   # number of observation points
obserr = 3.0 # observation error spread
H = rv.obs_operator(iX, jX, nv, iObs, jObs, vObs)
obs = np.dot(H, Xt) + np.array([2.0, -2.0])

iout = np.array([70])
jout = np.array([55])
vout = np.array([0])
Hout = rv.obs_operator(iX, jX, nv, iout, jout, vout)

##Run filter
Xa = np.zeros((4, nens, ni*nj*nv))
Xa[0, :, :] = Xb
Xa[1, :, :] = DA.filter_update(ni, nj, nv, Xb, iX, jX, H, iObs, jObs, vObs, obs, obserr, 0, np.arange(1, 2), 'EnSRF')
Xa[2, :, :] = DA.filter_update(ni, nj, nv, Xb, iX, jX, H, iObs, jObs, vObs, obs, obserr, 0, np.arange(1, 9), 'EnSRF')
Xa[3, :, :] = DA.filter_update(ni, nj, nv, Xb, iX, jX, H, iObs, jObs, vObs, obs, obserr, 0, np.arange(1, 2), 'PF')

##plot
plt.switch_backend('Agg')
plt.figure(figsize=(7, 7))
ii, jj = np.mgrid[0:ni, 0:nj]
cmap = [plt.cm.jet(m) for m in np.linspace(0.2, 0.8, nens)]

for i in range(4):
  ax = plt.subplot(2, 2, i+1)
  for m in range(nens):
    u, v = rv.X2uv(ni, nj, Xa[i, m, :])
    ax.contour(ii, jj, u, (-15, 15), colors=[cmap[m][0:3]], linewidths=2)
  # u, v = rv.X2uv(ni, nj, np.mean(Xa[i, :, :], axis=0))
  # ax.contour(ii, jj, u, (-15, 15), colors='r', linewidths=3)
  ut, vt = rv.X2uv(ni, nj, Xt)
  ax.contour(ii, jj, ut, (-15, 15), colors='k', linewidths=3)
  ax.plot(iObs, jObs, 'k+', markersize=10, markeredgewidth=2)
  ax.plot(iout, jout, 'kx', markersize=10, markeredgewidth=2)
  ax.set_aspect('equal', 'box')
  ax.set_xlim(43, 83)
  ax.set_ylim(43, 83)
  ax.set_xticks(np.arange(43, 84, 10))
  ax.set_xticklabels(np.arange(-20, 21, 10))
  ax.set_yticks(np.arange(43, 84, 10))
  ax.set_yticklabels(np.arange(-20, 21, 10))
  ax.tick_params(labelsize=12)

plt.savefig('Fig01.pdf')
