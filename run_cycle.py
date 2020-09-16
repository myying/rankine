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
nt = 13

### Rankine Vortex definition, forecast model
Rmw = 5    # radius of maximum wind
Vmax = 20   # maximum wind speed
Vout = 0    # wind speed outside of vortex
iStorm = 64 # location of vortex in i, j
jStorm = 64
nens = 20
Csprd = 6
bkg_err = 1e-4
gen_rate_ens = np.random.uniform(1.0, 1.0, nens)

filter_kind = sys.argv[1] #'NoDA'
ns = int(sys.argv[2])  ##number of scales
localize_cutoff = 30
obserr = 5.0
obs_intv = 1
cycle_start = 1
cycle_end = nt
cycle_period = 7200
casename = filter_kind+'_s{}'.format(ns)

##initial ensemble state
np.random.seed(0)
X_bkg = rv.make_background_flow(ni, nj, nv, dx, ampl=1e-4)
iX, jX = rv.make_coords(ni, nj)
X = np.zeros((ni*nj*nv, nens, 2, nt))  ###ensemble state 0:prior 1:posterior

iStorm_ens = np.zeros(nens)
jStorm_ens = np.zeros(nens)
for m in range(nens):
  iStorm_ens[m] = iStorm + np.random.normal(0, 1) * Csprd
  jStorm_ens[m] = jStorm + np.random.normal(0, 1) * Csprd
  X[:, m, 0, 0] = rv.make_state(ni, nj, nv, iStorm_ens[m], jStorm_ens[m], Rmw, Vmax, Vout)
  X[:, m, 0, 0] += X_bkg + rv.make_background_flow(ni, nj, nv, dx, ampl=bkg_err)

inflation = np.ones((ni*nj*nv, 2, ns, nt))  ###adaptive inflation field 0: inf_mean, 1: inf_sd
inflation[:, 0, :, :] = 1.0
inflation[:, 1, :, :] = 1.0

Xt = np.load(outdir+'truth_state.npy')
obs = np.load(outdir+'obs.npy')
iObs = np.load(outdir+'obs_i.npy')
jObs = np.load(outdir+'obs_j.npy')
vObs = np.load(outdir+'obs_v.npy')

loc = np.zeros((2, nens+1, 2, nt))  ###storm location 0:x 1:y; 0:prior 1:posterior
wind = np.zeros((nens+1, 2, nt))    ###max wind speed

for n in range(nt):
  print(n)

  if filter_kind != "NoDA" and n%obs_intv == 0 and n>=cycle_start and n<=cycle_end:
    ##DA update
    print('running '+filter_kind+' update')
    H = rv.obs_operator(iX, jX, nv, iObs[n, :], jObs[n, :], vObs[n, :])
    krange = np.arange(2, 2*ns+1, 2)
    Xb = X[:, :, 0, n].T
    infb = inflation[:, :, :, n]
    Xa, infa = DA.filter_update(ni, nj, nv, Xb, iX, jX, H, iObs[n, :], jObs[n, :], vObs[n, :], obs[n, :], obserr, localize_cutoff, infb, krange, filter_kind)
    X[:, :, 1, n] = Xa.T
    inflation[:, :, :, n] = infa
  else:
    X[:, :, 1, n] = X[:, :, 0, n]

  ##diagnose
  for i in range(2):
    u, v = rv.X2uv(ni, nj, X[:, :, i, n])
    zeta = rv.uv2zeta(u, v, dx)
    for m in range(nens):
      loc[0, m, i, n], loc[1, m, i, n] = rv.get_center_ij(u[:, :, m], v[:, :, m], dx)
      wind[m, i, n] = rv.get_max_wind(u[:, :, m], v[:, :, m])
    um = np.mean(u, axis=2)
    vm = np.mean(v, axis=2)
    loc[0, nens, i, n], loc[1, nens, i, n] = rv.get_center_ij(um, vm, dx)
    wind[nens, i, n] = rv.get_max_wind(um, vm)

  ##model forecast
  if n < nt-1:
    for m in range(nens):
      X[:, m, 0, n+1] = rv.advance_time(ni, nj, X[:, m, 1, n], dx, int(cycle_period/dt), dt, gen_rate_ens[m])

  np.save(outdir+casename+'_ens.npy', X)
  np.save(outdir+casename+'_ij.npy', loc)
  np.save(outdir+casename+'_wind.npy', wind)
  np.save(outdir+casename+'_inflation.npy', inflation)
