#!/usr/bin/env python
import numpy as np
import sys
import os
from rankine_vortex import *
from obs_def import *
from data_assimilation import *
from multiscale import *
from config import *

if (len(sys.argv)<6):
    print("usage: run_single.py filter_kind ns nens loc_sprd obs_r_vortex")
    exit()

nrealize = 10

filter_kind = sys.argv[1] #'EnSRF'
ns = int(sys.argv[2])
nens = int(sys.argv[3]) # ensemble size
loc_sprd = int(sys.argv[4])
obs_r_vortex = int(sys.argv[5])
local_cutoff = 0  ##no localization
krange = get_krange(ns)
krange_obs = (1,)
run_alignment = True

casename = '{}_s{}_N{}_l{}_r{}'.format(filter_kind, ns, nens, loc_sprd, obs_r_vortex)

if os.path.exists(outdir+'single/'+casename+'.npy'):
    state_err = np.load(outdir+'single/'+casename+'.npy')
else:
    state_err = np.zeros(nrealize)
    state_err[:] = np.nan

##truth
Xt = gen_vortex(ni, nj, nv, Vmax, Rmw, 0)

for realize in range(nrealize):
    np.random.seed(realize)
    if realize%100 == 0:
        print(realize)

    ##Prior ensemble
    Xb = np.zeros((ni, nj, nv, nens))
    for m in range(nens):
        Xb[:, :, :, m] = gen_vortex(ni, nj, nv, Vmax, Rmw, loc_sprd)

    ###Observation
    nobs = 2
    Ymask = np.ones(nobs)
    Yloc = np.zeros((3, nobs))
    th = np.random.uniform(0, 360)*np.pi/180*np.array([1, 1])
    Yloc[0, :] = 0.5*ni + obs_r_vortex*np.sin(th)
    Yloc[1, :] = 0.5*nj + obs_r_vortex*np.cos(th)
    Yloc[2, :] = np.array([0, 1])
    Yo = obs_interp2d(Xt, Yloc) + obs_err_std * np.random.normal(0, 1, nobs)

    ##Run filter
    Xa = Xb.copy()
    Xa = filter_update(Xb, Yo, Ymask, Yloc, filter_kind,
                       obs_err_std*np.ones(ns), np.ones(ns).astype(int), local_cutoff*np.ones(ns),
                       krange, krange_obs, run_alignment)

    ###Diagnose
    ###domain-averaged (near storm region) state (u,v) error:
    state_err[realize] = rmse(Xa, Xt)
    np.save(outdir+'single/'+casename+'.npy', state_err)

    ###intensity track
    # for m in range(nens):
    #     u, v = rv.X2uv(ni, nj, Xa[m, :])
    #     w = rv.get_max_wind(u, v)
    #     i, j = rv.get_center_ij(u, v, dx)
    #     feature_error[realize, 0, m] = np.abs(w-wt)
    #     feature_error[realize, 1, m] = 0.5*(np.abs(i-it)+np.abs(j-jt))
    # um, vm = rv.X2uv(ni, nj, np.mean(Xa, axis=0))
    # wm = rv.get_max_wind(um, vm)
    # im, jm = rv.get_center_ij(um, vm, dx)
    # feature_error[realize, 0, nens] = np.abs(wm-wt)
    # feature_error[realize, 1, nens] = 0.5*(np.abs(im-it)+np.abs(jm-jt))
    #np.save(outdir+'feature_error/'+casename+'.npy', feature_error)

