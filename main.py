#!/usr/bin/env python
import numpy as np
import rankine_vortex as rv
import graphics as g
import matplotlib.pyplot as plt
import data_assimilation as DA
import sys

np.random.seed(0)  # fix random number seed, make results predictable

ni = 128  # number of grid points i, j directions
nj = 128
nv = 2   # number of variables, (u, v)
nens = 200  # ensemble size
nens_show = 20

### Rankine Vortex definition, truth
Rmw = 15    # radius of maximum wind
Vmax = 50   # maximum wind speed
Vout = 10    # wind speed outside of vortex
iStorm = 63 # location of vortex in i, j
jStorm = 63

filter_kind = sys.argv[1] #'EnSRF'
localize_cutoff = 30

iX, jX = rv.make_coords(ni, nj)
Xt = rv.make_state(ni, nj, nv, iStorm, jStorm, Rmw, Vmax, Vout)

##Prior ensemble
Csprd = 10
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

###observations
# iObs = np.random.uniform(0, nj, size=nobs*nv)
# iObs = np.array([50, 50])
# jObs = np.array([75, 75])
# vObs = np.array([0, 1])
iObs = np.array([50])
jObs = np.array([75])
vObs = np.array([0])
nobs = iObs.size   # number of observation points
obserr = 1 # observation error spread
L = rv.location_operator(iX, jX, iObs, jObs)
H = rv.obs_operator(iX, jX, nv, iObs, jObs, vObs)
obs = np.matmul(H, Xt) + np.random.normal(0.0, obserr, nobs)

##filter
Xa = Xb.copy()
if filter_kind == 'EnSRF':
  Xa = DA.EnSRF(ni, nj, nv, Xb, iX, jX, H, iObs, jObs, vObs, obs, obserr, localize_cutoff)
if filter_kind == 'PF':
  Xa = DA.PF(ni, nj, nv, Xb, iX, jX, H, iObs, jObs, vObs, obs, obserr, localize_cutoff)
if filter_kind == 'EnSRF_MSA':

  Xa = DA.EnSRF(ni, nj, nv, Xb, iX, jX, H, iObs, jObs, vObs, obs, obserr, localize_cutoff)

##diagnose & plot
# plt.switch_backend('Agg')
plt.figure(figsize=(5, 5))
ax = plt.subplot(1, 1, 1)
# ax.scatter(iStorm_ens,jStorm_ens, s=3, color='.7')
cmap = [plt.cm.jet(x) for x in np.linspace(0, 1,nens_show)]
for n in range(nens_show):
  # ax.scatter(iStorm_ens[n],jStorm_ens[n], s=40, color=[cmap[n][0:3]])
  g.plot_contour(ax,ni,nj, Xa[n, :], [cmap[n][0:3]], 1)
g.plot_contour(ax, ni, nj, Xt, 'black', 3)
ax.plot(iObs, jObs, 'kx')
g.set_axis(ax,ni,nj)
ax.tick_params(labelsize=15)
g.output_ens('1.nc', ni, nj, Xa, Xt)

# plt.savefig('ens.png', dpi=100)
plt.show()


