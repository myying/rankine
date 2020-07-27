#!/usr/bin/env python
import numpy as np
import rankine_vortex as rv
import data_assimilation as DA
import sys

outdir = '/Users/mying/work/rankine/cycle/'
ni = 128  # number of grid points i, j directions
nj = 128
nv = 2   # number of variables, (u, v)
dx = 9000
dt = 300
nt = 5
diss = 3*1e3

### Rankine Vortex definition, truth
Rmw = 5    # radius of maximum wind
Vmax = 50   # maximum wind speed
Vout = 0    # wind speed outside of vortex
iStorm = 83 # location of vortex in i, j
jStorm = 53
Csprd = 8
nens = 40
filter_kind = sys.argv[1] #'NoDA'
ns = int(sys.argv[2])
localize_cutoff = 50
obserr = 3.0
cycle_period = 3600*1
#diss_ens = 3e2*np.ones(nens) ##+ np.random.uniform(-2, 4, (nens,))

##initial ensemble state
np.random.seed(0)
X_bkg = rv.make_background_flow(ni, nj, nv, dx, ampl=1e-4)
iX, jX = rv.make_coords(ni, nj)
X = np.zeros((ni*nj*nv, nens, nt+1))
loc = np.zeros((2, nens, nt+1))
wind = np.zeros((nens, nt+1))

iStorm_ens = np.zeros(nens)
jStorm_ens = np.zeros(nens)
for m in range(nens):
  iStorm_ens[m] = iStorm + np.random.normal(0, 1) * Csprd
  jStorm_ens[m] = jStorm + np.random.normal(0, 1) * Csprd
  X[:, m, 0] = rv.make_state(ni, nj, nv, iStorm_ens[m], jStorm_ens[m], Rmw, Vmax, Vout)
  X[:, m, 0] += X_bkg + rv.make_background_flow(ni, nj, nv, dx, ampl=5e-5)

Xt = np.load(outdir+'truth_state.npy')
obs = np.load(outdir+'obs.npy')
iObs = np.load(outdir+'obs_i.npy')
jObs = np.load(outdir+'obs_j.npy')
vObs = np.load(outdir+'obs_v.npy')

for n in range(nt):
  print(n)

  ##DA update
  H = rv.obs_operator(iX, jX, nv, iObs[n, :], jObs[n, :], vObs[n, :])
  krange = np.arange(2, 2*ns+1, 2)
  Xa = DA.filter_update(ni, nj, nv, X[:, :, n].T, iX, jX, H, iObs[n, :], jObs[n, :], vObs[n, :], obs[n, :], obserr, localize_cutoff, krange, filter_kind)
  X[:, :, n] = Xa.T

  ##diagnose
  u, v = rv.X2uv(ni, nj, X[:, :, n])
  zeta = rv.uv2zeta(u, v, dx)
  for m in range(nens):
    loc[0, m, n], loc[1, m, n] = rv.get_center_ij(u[:, :, m], v[:, :, m], dx)
    wind[m, n] = rv.get_max_wind(u[:, :, m], v[:, :, m])

  ##model forecast
  if n < nt-1:
    X[:, :, n+1] = rv.advance_time(ni, nj, X[:, :, n], dx, int(cycle_period/dt), dt, diss)

casename = filter_kind+'_s{}'.format(ns)
np.save(outdir+casename+'_ens.npy', X)
np.save(outdir+casename+'_ij.npy', loc)
np.save(outdir+casename+'_wind.npy', wind)
