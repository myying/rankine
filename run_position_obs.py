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
dirname = 'position_obs/{:04d}'.format(realize)
if not os.path.exists(outdir+dirname):
    os.makedirs(outdir+dirname)

###Observation
Ymask = np.ones(2)
Yloc = np.zeros((3, 2))
Yloc[2, :] = np.array([-1, -1])
obs_err_std = 0.5
if os.path.isfile(outdir+dirname+'/Yo.npy'):
    Yo = np.load(outdir+dirname+'/Yo.npy')
else:
    Yo = vortex_center(Xt) + obs_err_std * np.random.normal(0, 1, 2)
    np.save(outdir+dirname+'/Yo.npy', Yo)

for loc_sprd in (1, 2, 3, 4, 5):
    loc_bias = 0
    nens = 20
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

    ##Run filter
    for ns in (1, 2, 3, 4, 5, 6, 7):
        if not os.path.isfile(outdir+dirname+scenario+'/EnSRF_s{}.npy'.format(ns)):
            Xa = filter_update(Xb, Yo, Ymask, Yloc, 'EnSRF', obs_err_std*np.ones(ns), 0.0*np.ones(ns), get_krange(ns), (1,), run_alignment=True)
            err = diagnose(Xa, Xt)
            np.save(outdir+dirname+scenario+'/EnSRF_s{}.npy'.format(ns), err)

    if not os.path.isfile(outdir+dirname+scenario+'/PF.npy'):
        Xa = filter_update(Xb, Yo, Ymask, Yloc, 'PF', obs_err_std*np.ones(1), 0.0*np.ones(1), get_krange(1), (1,), run_alignment=False)
        err = diagnose(Xa, Xt)
        np.save(outdir+dirname+scenario+'/PF.npy', err)

