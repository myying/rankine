#!/usr/bin/env python
import numpy as np
import rankine_vortex as rv
import data_assimilation as DA
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
obserr = 5.0
obs_intv = 1
cycle_start = 3
cycle_end = 11
cycle_period = 7200

nens = 20
Csprd = 5
bkg_err = 1e-4
localize_cutoff = 30

casename = sys.argv[1]
nrealize = int(sys.argv[2])
filter_kind = sys.argv[3] #'NoDA'
ns = int(sys.argv[4])  ##number of scales

np.random.seed(nrealize)
outdir = '/glade/scratch/mying/rankine/cycle/'+casename+'/{:03d}/'.format(nrealize)

gen_rate = np.load(outdir+'truth_gen_rate.npy')
if casename == 'perfect_model':
  gen_rate_ens = np.ones(nens) * gen_rate
if casename == 'imperfect_model':
  gen_rate_ens = np.random.uniform(max(0.0, gen_rate-1.0), min(2.0, gen_rate+0.5), nens)

##initial ensemble state
X_bkg = rv.make_background_flow(ni, nj, nv, dx, ampl=1e-4)
iX, jX = rv.make_coords(ni, nj)
Xens = np.zeros((ni*nj*nv, nens, 2, nt))  ###ensemble state 0:prior 1:posterior
loc_ens = np.zeros((2, nens+1, 2, nt)) ##i(0) j(1); nens and mean; prior(0) post(1), num cycles
wind_ens = np.zeros((nens+1, 2, nt))
state_err_sprd = np.zeros((2, 2, nt))  ##err(0) spread(1); prior(0) post(1); num cycles

iStorm_ens = np.zeros(nens)
jStorm_ens = np.zeros(nens)
for m in range(nens):
  iStorm_ens[m] = iStorm + np.random.normal(0, 1) * Csprd
  jStorm_ens[m] = jStorm + np.random.normal(0, 1) * Csprd
  Xens[:, m, 0, 0] = rv.make_state(ni, nj, nv, iStorm_ens[m], jStorm_ens[m], Rmw, Vmax, Vout)
  Xens[:, m, 0, 0] += X_bkg + rv.make_background_flow(ni, nj, nv, dx, ampl=bkg_err)

inflation = np.ones((ni*nj*nv, 2, ns, nt))  ###adaptive inflation field 0: inf_mean, 1: inf_sd
inflation[:, 0, :, :] = 1.0
inflation[:, 1, :, :] = 1.0

##load truth and obs
X = np.load(outdir+'truth_state.npy')
loc = np.load(outdir+'truth_ij.npy')
wind = np.load(outdir+'truth_wind.npy')
obs = np.load(outdir+'obs.npy')
iObs = np.load(outdir+'obs_i.npy')
jObs = np.load(outdir+'obs_j.npy')
vObs = np.load(outdir+'obs_v.npy')

for n in range(nt):
  # print(n)

  if filter_kind != "NoDA" and n%obs_intv == 0 and n>=cycle_start and n<=cycle_end:
    ##DA update
    print('running '+filter_kind+' update')
    H = rv.obs_operator(iX, jX, nv, iObs[n, :], jObs[n, :], vObs[n, :])
    krange = np.arange(2, 2*ns+1, 2)
    Xb = Xens[:, :, 0, n].T
    infb = inflation[:, :, :, n]
    Xa, infa = DA.filter_update(ni, nj, nv, Xb, iX, jX, H, iObs[n, :], jObs[n, :], vObs[n, :], obs[n, :], obserr, localize_cutoff, infb, krange, filter_kind, run_inflation=True, run_alignment=True)
    Xens[:, :, 1, n] = Xa.T
    inflation[:, :, :, n] = infa
  else:
    Xens[:, :, 1, n] = Xens[:, :, 0, n]

  ##diagnose
  for i in range(2):  ##prior and posterior
    ##physics space rmse
    u, v = rv.X2uv(ni, nj, Xens[:, :, i, n])
    um, vm = rv.X2uv(ni, nj, np.mean(Xens[:, 0:nens, i, n], axis=1))
    ut, vt = rv.X2uv(ni, nj, X[:, n])
    sq_err = (um - ut)**2 + (vm - vt)**2
    sq_sprd = np.zeros((ni, nj))
    for m in range(nens):
      sq_sprd += (u[:, :, m] - um)**2 + (v[:, :, m] - vm)**2
    loc_i = int(loc[0, n])
    loc_j = int(loc[1, n])
    buff = 10
    state_err_sprd[0, i, n] = np.sqrt(np.mean(sq_err[loc_i-buff:loc_i+buff, loc_j-buff:loc_j+buff]))
    state_err_sprd[1, i, n] = np.sqrt(np.sum(sq_sprd[loc_i-buff:loc_i+buff, loc_j-buff:loc_j+buff])/(nens-1))

    ##feature space: position and intensity, size
    zeta = rv.uv2zeta(u, v, dx)
    for m in range(nens):
      loc_ens[0, m, i, n], loc_ens[1, m, i, n] = rv.get_center_ij(u[:, :, m], v[:, :, m], dx)
      wind_ens[m, i, n] = rv.get_max_wind(u[:, :, m], v[:, :, m])
    loc_ens[0, nens, i, n], loc_ens[1, nens, i, n] = rv.get_center_ij(um, vm, dx)
    wind_ens[nens, i, n] = rv.get_max_wind(um, vm)

  ##model forecast
  if n < nt-1:
    for m in range(nens):
      Xens[:, m, 0, n+1] = rv.advance_time(ni, nj, Xens[:, m, 1, n], dx, int(cycle_period/dt), dt, gen_rate_ens[m])

output = outdir+filter_kind+'_s{}'.format(ns)
np.save(output+'_ens.npy', Xens)
np.save(output+'_inflation.npy', inflation)
np.save(output+'_gen_rate.npy', gen_rate_ens)
np.save(output+'_state_err_sprd.npy', state_err_sprd)
np.save(output+'_loc_ens.npy', loc_ens)
np.save(output+'_wind_ens.npy', wind_ens)
