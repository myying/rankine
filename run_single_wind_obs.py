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

##truth
Xt = gen_vortex(ni, nj, nv, Vmax, Rmw)

np.random.seed(realize)
dirname = 'single_wind_obs/{:04d}'.format(realize)
if not os.path.exists(outdir+dirname):
    os.makedirs(outdir+dirname)

###Observation
Ymask = np.ones(2)
if os.path.isfile(outdir+dirname+'/Yo.npy'):
    Yo = np.load(outdir+dirname+'/Yo.npy')
    Yloc = np.load(outdir+dirname+'/Yloc.npy')
else:
    Yloc = np.zeros((3, 2))
    obs_r = np.random.uniform(0, 10)*np.ones(2)
    obs_th = np.random.uniform(0, 360)*np.pi/180*np.ones(2)
    Yloc[0, :] = 0.5*ni + obs_r*np.sin(obs_th)
    Yloc[1, :] = 0.5*nj + obs_r*np.cos(obs_th)
    Yloc[2, :] = np.array([0, 1])
    Yo = obs_interp2d(Xt, Yloc) + obs_err_std * np.random.normal(0, 1, 2)
    np.save(outdir+dirname+'/Yo.npy', Yo)
    np.save(outdir+dirname+'/Yloc.npy', Yloc)

for loc_sprd in (1, 2, 3, 4, 5):
    for loc_bias in (0, 5):
        for nens in (20,):
            scenario = "/Lbias{}/Lsprd{}/N{}".format(loc_bias, loc_sprd, nens)
            if not os.path.exists(outdir+dirname+scenario):
                os.makedirs(outdir+dirname+scenario)

            ##Prior ensemble
            Xb = np.zeros((ni, nj, nv, nens))
            for m in range(nens):
                Xb[:, :, :, m] = gen_vortex(ni, nj, nv, Vmax, Rmw, loc_sprd, loc_bias)
            if not os.path.isfile(outdir+dirname+scenario+'/NoDA.npy'):
                err = diagnose(Xb, Xt)
                np.save(outdir+dirname+scenario+'/NoDA.npy', err)

            ##Run filter with MSA:
            for ns in (1, 2, 3, 4, 5, 6, 7):
                if not os.path.isfile(outdir+dirname+scenario+'/EnSRF_s{}.npy'.format(ns)):
                    Xa = filter_update(Xb, Yo, Ymask, Yloc, 'EnSRF', obs_err_std*np.ones(ns), np.zeros(ns), np.ones(ns), get_krange(ns), (1,), run_alignment=True)
                    err = diagnose(Xa, Xt)
                    np.save(outdir+dirname+scenario+'/EnSRF_s{}.npy'.format(ns), err)

            ##particle filter solution
            if not os.path.isfile(outdir+dirname+scenario+'/PF.npy'):
                Xa = filter_update(Xb, Yo, Ymask, Yloc, 'PF', obs_err_std*np.ones(1), np.zeros(1), np.ones(1), get_krange(1), (1,), run_alignment=False)
                err = diagnose(Xa, Xt)
                np.save(outdir+dirname+scenario+'/PF.npy', err)

