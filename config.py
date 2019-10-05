#!/usr/bin/env python
import numpy as np
import rankine_vortex as rv

np.random.seed(1)  # fix random number seed, make results predictable

ni = 41  # number of grid points i, j directions
nj = 41
nv = 2   # number of variables, (u, v)
nens = 1000  # ensemble size
nens_show = 8

### Rankine Vortex definition, truth
Rmw = 5    # radius of maximum wind
Vmax = 30   # maximum wind speed
Vout = 0    # wind speed outside of vortex
iStorm = 20 # location of vortex in i, j
jStorm = 20
loc_sprd = 1.5

nobs = 500   # number of observations
obserr = 1.0 # observation error spread
localize_cutoff = 100  # localization cutoff distance (taper to zero)

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
  # iD, jD = al.random_vector(ni, nj, np.array([0, 0]), 20, 3)
  # for v in range(nv):
  #   Xb[n, v*ni*nj:(v+1)*ni*nj] = al.deformation(ni, nj, Xb[n, v*ni*nj:(v+1)*ni*nj], iD, jD)

###observations (radial velocity)
iObs = np.random.uniform(0, ni, size=nobs)
jObs = np.random.uniform(0, nj, size=nobs)
L = rv.location_operator(iX, jX, iObs, jObs)
iSite = 2
jSite = 2
H = rv.obs_operator(iX, jX, nv, iObs, jObs, iSite, jSite)
obs = np.matmul(H, Xt) + np.random.normal(0.0, obserr, nobs)

