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

nens = int(sys.argv[1]) # ensemble size
loc_sprd = int(sys.argv[2])


##truth
Xt = gen_vortex(ni, nj, nv, Vmax, Rmw, 0)

for realize in range(nrealize):
    np.random.seed(realize)
    dirname = 'single_obs/N{}/L{}/{:04d}'.format(nens, loc_sprd, realize+1)
    print(outdir+dirname)
    if not os.path.exists(outdir+dirname):
        os.makedirs(outdir+dirname)

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

    np.save(outdir+dirname+'/Yo.npy', Yo)
    np.save(outdir+dirname+'/Yloc.npy', Yloc)

    err = err_diag(Xb, Xt)
    np.save(outdir+dirname+'/NoDA.npy', err)

    ##Run filter
    for ns in range(1, 8):
        Xa = filter_update(Xb, Yo, Ymask, Yloc, 'EnSRF', obs_err_std*np.ones(ns), 0.0*np.ones(ns), get_krange(ns), (1,), run_alignment=True)
        err = err_diag(Xa, Xt)
        np.save(outdir+dirname+'/EnSRF_s{}.npy'.format(ns), err)
        # np.save(outdir+'single_obs/'+'N{}/L{}/{:04d}/EnSRF_s{}.npy'.format(nens, loc_sprd, realize+1, ns), Xa)

    Xa = filter_update(Xb, Yo, Ymask, Yloc, 'PF', obs_err_std*np.ones(1), 0.0*np.ones(1), get_krange(1), (1,), run_alignment=False)
    err = err_diag(Xa, Xt)
    np.save(outdir+dirname+'/PF.npy', err)


