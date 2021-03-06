#!/usr/bin/env python
import numpy as np
import rankine_vortex as rv
import sys

ni = 128  # number of grid points i, j directions
nj = 128
nv = 2   # number of variables, (u, v)
dx = 9000
dt = 300
nt = 19
Rmw = 5    # radius of maximum wind
Vmax = 20   # maximum wind speed
Vout = 0    # wind speed outside of vortex
iStorm = 64 # location of vortex in i, j
jStorm = 64
obs_range = 25
nobs = 50
obserr = 5.0
cycle_period = 7200

casename = sys.argv[1]
nrealize = int(sys.argv[2])
np.random.seed(nrealize)
if casename == 'perfect_model':
  gen_rate = np.random.uniform(0.0, 2.0)
if casename == 'imperfect_model':
  gen_rate = np.random.uniform(1.0, 1.5)
outdir = '/glade/scratch/mying/rankine/cycle/'+casename+'/{:03d}/'.format(nrealize)

iX, jX = rv.make_coords(ni, nj)
X = np.zeros((ni*nj*nv, nt))
loc = np.zeros((2, nt))
wind = np.zeros(nt)
size = np.zeros(nt)
obs = np.zeros((nt, nobs*nv))
iObs = np.zeros((nt, nobs*nv))
jObs = np.zeros((nt, nobs*nv))
vObs = np.zeros((nt, nobs*nv))

##initial state
X_bkg = rv.make_background_flow(ni, nj, nv, dx, ampl=1e-4)
X[:, 0] = X_bkg + rv.make_state(ni, nj, nv, iStorm, jStorm, Rmw, Vmax, Vout)

for n in range(nt):
  # print(n)
  u, v = rv.X2uv(ni, nj, X[:, n])
  zeta = rv.uv2zeta(u, v, dx)
  loc[0, n], loc[1, n] = rv.get_center_ij(u, v, dx)
  wind[n] = rv.get_max_wind(u, v)
  size[n] = rv.get_size(u, v, loc[0, n], loc[1, n])

  for p in range(nobs):
    for v in range(nv):
      iObs[n, p*nv+v] = np.random.uniform(-obs_range, obs_range) + loc[0, n]
      jObs[n, p*nv+v] = np.random.uniform(-obs_range, obs_range) + loc[1, n]
      vObs[n, p*nv+v] = v
  H = rv.obs_operator(iX, jX, nv, iObs[n, :], jObs[n, :], vObs[n, :])
  obs[n, :] = np.dot(H, X[:, n]) + np.random.normal(0.0, obserr, (nobs*nv,))

  if n < nt-1:
    X[:, n+1] = rv.advance_time(ni, nj, X[:, n], dx, int(cycle_period/dt), dt, gen_rate)

np.save(outdir+'truth_state.npy', X)
np.save(outdir+'truth_loc.npy', loc)
np.save(outdir+'truth_wind.npy', wind)
np.save(outdir+'truth_size.npy', size)
np.save(outdir+'obs.npy', obs)
np.save(outdir+'obs_i.npy', iObs)
np.save(outdir+'obs_j.npy', jObs)
np.save(outdir+'obs_v.npy', vObs)
np.save(outdir+'truth_gen_rate.npy', gen_rate)
