#!/usr/bin/env python
import numpy as np
import rankine_vortex as rv
import data_assimilation as DA
import sys
import os

if (len(sys.argv)<7):
  print("usage: run_single.py filter_kind ns nens Csprd obsR obsErr")
  exit()

outdir = '/glade/scratch/mying/rankine/single/'
nrealize = 10000
r0 = 0

ni = 128  # number of grid points i, j directions
nj = 128
nv = 2   # number of variables, (u, v)
dx = 9000

### Rankine Vortex definition, truth
Rmw = 5    # radius of maximum wind
Vmax = 50   # maximum wind speed
Vout = 0    # wind speed outside of vortex
iStorm = 63 # location of vortex in i, j
jStorm = 63
iBias = 0
jBias = 0
Rsprd = 0
Vsprd = 0

filter_kind = sys.argv[1] #'EnSRF'
ns = int(sys.argv[2])
nens = int(sys.argv[3]) # ensemble size
Csprd = int(sys.argv[4])
obsR = int(sys.argv[5])
obserr = int(sys.argv[6]) # observation error spread

##truth
iX, jX = rv.make_coords(ni, nj)
Xt = rv.make_state(ni, nj, nv, iStorm, jStorm, Rmw, Vmax, Vout)
ut, vt = rv.X2uv(ni, nj, Xt)

##DA trials
casename = '{}_s{}_N{}_C{}_R{}_err{}'.format(filter_kind, ns, nens, Csprd, obsR, obserr)

if os.path.exists(outdir+casename+'.npy'):
  state_error = np.load(outdir+casename+'.npy')
else:
  state_error = np.zeros((nrealize, nens+1))
  state_error[:, :] = np.nan

for realize in range(nrealize):
  np.random.seed(realize)  # fix random number seed, make results predictable
  if realize%100 == 0:
    print(realize)

  ##Prior ensemble
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
  th = np.random.uniform(0, 360)*np.pi/180*np.array([1, 1])
  iObs = iStorm + obsR*np.sin(th)
  jObs = iStorm + obsR*np.cos(th)
  vObs = np.array([0, 1])
  nobs = iObs.size   # number of observation points
  H = rv.obs_operator(iX, jX, nv, iObs, jObs, vObs)
  obs = np.dot(H, Xt) + np.random.normal(0.0, obserr, nobs)

  ##Run filter
  Xa = Xb.copy()
  krange = np.arange(1, ns+1)
  infb = np.ones((ni*nj*nv, 2))
  Xa, infa = DA.filter_update(ni, nj, nv, Xb, iX, jX, H, iObs, jObs, vObs, obs, obserr, localize_cutoff=0, infb, krange, filter_kind, run_inflation=False, run_alignment=True)

  ###Diagnose
  ###domain-averaged (near storm region) state (u,v) error:
  for m in range(nens):
    u, v = rv.X2uv(ni, nj, Xa[m, :])
    square_err = (u-ut)**2 + (v-vt)**2
    state_error[realize, m] = np.sqrt(np.mean(square_err[iStorm-20:iStorm+20, jStorm-20:jStorm+20]))
  state_error[realize, nens] = np.sqrt(np.mean((np.mean(Xa, axis=0)-Xt)**2))
  np.save(outdir+casename+'.npy', state_error)
  # u, v = rv.X2uv(ni, nj, np.mean(Xa, axis=0))
  # square_err = (u-ut)**2 + (v-vt)**2
  # state_error = np.sqrt(np.mean(square_err))
  #state_error = np.sqrt(np.mean(square_err[iStorm-20:iStorm+20, jStorm-20:jStorm+20]))
  #print(state_error)


  ###intensity track
  # for m in range(nens):
  #   u, v = rv.X2uv(ni, nj, Xa[m, :])
  #   w = rv.get_max_wind(u, v)
  #   i, j = rv.get_center_ij(u, v, dx)
  #   feature_error[realize, 0, m] = np.abs(w-wt)
  #   feature_error[realize, 1, m] = 0.5*(np.abs(i-it)+np.abs(j-jt))
  # um, vm = rv.X2uv(ni, nj, np.mean(Xa, axis=0))
  # wm = rv.get_max_wind(um, vm)
  # im, jm = rv.get_center_ij(um, vm, dx)
  # feature_error[realize, 0, nens] = np.abs(wm-wt)
  # feature_error[realize, 1, nens] = 0.5*(np.abs(im-it)+np.abs(jm-jt))
  #np.save(outdir+'feature_error/'+casename+'.npy', feature_error)

