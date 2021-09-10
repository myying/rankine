#!/usr/bin/env python
import numpy as np
import sys
import os
from rankine_vortex import *
from obs_def import *
from data_assimilation import *
from multiscale import *
from config import *

realize = int(sys.argv[1])
nens = 200
loc_sprd = 0.1*Rmw
loc_bias = 0.0*Rmw

##truth
Xt = gen_vortex(ni, nj, nv, Vmax, Rmw)

np.random.seed(realize)
dirname = 'single_wind_obs/{:04d}'.format(realize+1)
if not os.path.exists(outdir+dirname):
    os.makedirs(outdir+dirname)

###Observation
Ymask = np.ones(2)
Yloc = np.zeros((3, 2))
##type1: wind obs at random location (obs_r, obs_th) vortex relative polar coords
obs_r = np.random.uniform(0, 10)*np.ones(2)
obs_th = np.random.uniform(0, 360)*np.pi/180*np.ones(2)
Yloc[0, :] = 0.5*ni + obs_r*np.sin(obs_th)
Yloc[1, :] = 0.5*nj + obs_r*np.cos(obs_th)
Yloc[2, :] = np.array([0, 1])
Yo = obs_interp2d(Xt, Yloc) + obs_err_std * np.random.normal(0, 1, 2)

##type2: position obs of vortex center
# obs_err_std = 0.5
# Yloc[2, :] = np.array([-1, -1])
# Yo = vortex_center(Xt) + obs_err_std * np.random.normal(0, 1, 2)

np.save(outdir+dirname+'/Yo.npy', Yo)
np.save(outdir+dirname+'/Yloc.npy', Yloc)

for loc_sprd in (1, 2, 3, 4, 5):
    loc_bias = 0
    scenario = "/Lsprd{}.Lbias{}".format(loc_sprd, loc_bias)
    if not os.path.exists(outdir+dirname+scenario):
        os.makedirs(outdir+dirname+scenario)

    ##Prior ensemble
    Xb = np.zeros((ni, nj, nv, nens))
    for m in range(nens):
        Xb[:, :, :, m] = gen_vortex(ni, nj, nv, Vmax, Rmw, loc_sprd, loc_bias)

    err = diagnose(Xb, Xt)
    np.save(outdir+dirname+scenario+'/NoDA.npy', err)

    ##Run filter
    for ns in (1,):
        Xa = filter_update(Xb, Yo, Ymask, Yloc, 'EnSRF', obs_err_std*np.ones(ns), 0.0*np.ones(ns), get_krange(ns), (1,), run_alignment=True)
        err = diagnose(Xa, Xt)
        np.save(outdir+dirname+scenario+'/EnSRF_s{}.npy'.format(ns), err)
        # np.save(outdir+'single_obs/'+'N{}/L{}/{:04d}/EnSRF_s{}.npy'.format(nens, loc_sprd, realize+1, ns), Xa)

    Xa = filter_update(Xb, Yo, Ymask, Yloc, 'PF', obs_err_std*np.ones(1), 0.0*np.ones(1), get_krange(1), (1,), run_alignment=False)
    err = diagnose(Xa, Xt)
    np.save(outdir+dirname+scenario+'/PF.npy', err)


