#!/usr/bin/env python
import numpy as np
import rankine_vortex as rv
import graphics as g
import matplotlib.pyplot as plt
import data_assimilation as DA

np.random.seed(1)  # fix random number seed, make results predictable

ni = 128  # number of grid points i, j directions
nj = 128
nv = 2   # number of variables, (u, v)
nens = 100  # ensemble size
nens_show = 10

### Rankine Vortex definition, truth
Rmw = 5    # radius of maximum wind
Vmax = 30   # maximum wind speed
Vout = 0    # wind speed outside of vortex
iStorm = 20 # location of vortex in i, j
jStorm = 20
loc_sprd = 0.5

nobs = 1   # number of observations
obserr = 1.0 # observation error spread
localize_cutoff = 9999  # localization cutoff distance (taper to zero)

iX, jX = rv.make_coords(ni, nj)

Xt = rv.make_state(ni, nj, nv, iStorm, jStorm, Rmw, Vmax, Vout)

##Prior ensemble
Xb = np.zeros((nens, ni*nj*nv))
Csprd = loc_sprd*Rmw
iBias = 0
jBias = 0
Rsprd = 0
Vsprd = 0
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
iObs = np.random.uniform(0, ni, size=nobs)
jObs = np.random.uniform(0, nj, size=nobs)
L = rv.location_operator(iX, jX, iObs, jObs)
iSite = 2
jSite = 2
H = rv.obs_operator(iX, jX, nv, iObs, jObs, iSite, jSite)
obs = np.matmul(H, Xt) + np.random.normal(0.0, obserr, nobs)

##filter
nens, nX = Xb.shape
x_in = 17
y_in = 29
iout = np.array([x_in])
jout = np.array([y_in])
H = rv.obs_operator(iX,jX,nv, iout, jout,iSite,jSite)
obs = np.matmul(H,Xt) + np.random.normal(0.0,obserr)
Xa = DA.LPF(ni,nj,nv,Xb,iX,jX, H, iout, jout, obs,obserr,localize_cutoff)

##diagnose & plot
plt.switch_backend('Agg')
plt.figure(figsize=(3, 3))
ax = plt.subplot(1, 1, 1)
# ax.scatter(iStorm_ens,jStorm_ens, s=3, color='.7')
g.plot_wind_contour(ax, ni, nj, Xt, 'black', 2)
cmap = [plt.cm.jet(x) for x in np.linspace(0, 1,nens_show)]
for n in range(nens_show):
  # ax.scatter(iStorm_ens[n],jStorm_ens[n], s=40, color=[cmap[n][0:3]])
  # g.plot_wind_contour(ax,ni,nj,Xb[n, :], [cmap[n][0:3]], 2)
  g.plot_wind_contour(ax,ni,nj, Xa[n, :], [cmap[n][0:3]], 2)
g.set_axis(ax,ni,nj)
# ax.tick_params(labelsize=15)

plt.savefig('ens.png', dpi=100)


