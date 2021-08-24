#!/usr/bin/env python
import numpy as np
import sys
import os
from rankine_vortex import *
from obs_def import *
from data_assimilation import *
from multiscale import *
from config import *

nrealize = 2000

nens = 20 # ensemble size
loc_sprd = int(sys.argv[1])


##truth
Xt = gen_vortex(ni, nj, nv, Vmax, Rmw, 0)
true_center = vortex_center(Xt)
true_intensity = vortex_intensity(Xt)

for realize in range(nrealize):
    np.random.seed(realize)
    if realize%100 == 0:
        print(realize)

    ##Prior ensemble
    Xb = np.zeros((ni, nj, nv, nens))
    for m in range(nens):
        Xb[:, :, :, m] = gen_vortex(ni, nj, nv, Vmax, Rmw, loc_sprd)

    ###Observation
    Ymask = np.ones(2)
    Yloc = np.zeros((3, 2))
    obs_r = np.random.uniform(0, 10)*np.ones(2)
    obs_th = np.random.uniform(0, 360)*np.pi/180*np.ones(2)
    Yloc[0, :] = 0.5*ni + obs_r*np.sin(obs_th)
    Yloc[1, :] = 0.5*nj + obs_r*np.cos(obs_th)
    Yloc[2, :] = np.array([0, 1])
    Yo = obs_interp2d(Xt, Yloc) + obs_err_std * np.random.normal(0, 1, 2)

    np.save(outdir+'single_obs/N{}/L{}/{:04d}/Yo.npy'.format(nens, loc_sprd, realize+1), Yo)
    np.save(outdir+'single_obs/N{}/L{}/{:04d}/Yloc.npy'.format(nens, loc_sprd, realize+1), Yloc)

    ##Run filter
    np.save(outdir+'single_obs/'+'N{}/L{}/{:04d}/NoDA.npy'.format(nens, loc_sprd, realize+1), Xb)

    for ns in range(1, 8):
        Xa = filter_update(Xb, Yo, Ymask, Yloc, 'EnSRF', obs_err_std*np.ones(ns), 0.0*np.ones(ns), get_krange(ns), (1,), run_alignment=True)
        np.save(outdir+'single_obs/'+'N{}/L{}/{:04d}/EnSRF_s{}.npy'.format(nens, loc_sprd, realize+1, ns), Xa)

    np.save(outdir+'single_obs/'+'N{}/L{}/{:04d}/PF.npy'.format(nens, loc_sprd, realize+1), Xa)

    ##Diagnose
    ###domain-averaged (near storm region) state (u,v) error:
    # for m in range(nens):
        # err[realize, m, 0] = np.sqrt(np.mean((Xa[:, :, :, m] - Xt)**2, axis=(0,1,2)))
        # err[realize, m, 1] = np.sqrt(np.mean((vortex_center(Xa[:, :, :, m]) - true_center)**2))
        # err[realize, m, 2] = np.sqrt(np.mean((vortex_intensity(Xa[:, :, :, m]) - true_intensity)**2))
    # err[realize, nens, 0] = rmse(Xa, Xt)
    # err[realize, nens, 1] = np.sqrt(np.mean((vortex_center(np.mean(Xa, axis=3)) - true_center)**2))
    # err[realize, nens, 2] = np.sqrt(np.mean((vortex_intensity(np.mean(Xa, axis=3)) - true_intensity)**2))


