#!/usr/bin/env python
import numpy as np
import sys
import os
from rankine_vortex import *
from obs_def import *
from data_assimilation import *
from multiscale import *
from config import *

nrealize = 200

nens = 20 # ensemble size
network_type = int(sys.argv[1])
loc_sprd = int(sys.argv[2])
local_cutoff = int(sys.argv[3])

if network_type==1:  ##global network
    nobs = 4000
    obs_range = 200
if network_type==2:  ##targeted network
    nobs = 16000
    obs_range = 30

##truth
Xt = gen_random_flow(ni, nj, nv, dx, Vbg, -3) + gen_vortex(ni, nj, nv, Vmax, Rmw, 0)
true_center = vortex_center(Xt)

for realize in range(nrealize):
    np.random.seed(realize)
    dirname = 'full_network/network_type{}/L{}/ROI{}/{:04d}'.format(network_type, loc_sprd, local_cutoff, realize+1)
    print(outdir+dirname)
    if not os.path.exists(outdir+dirname):
        os.makedirs(outdir+dirname)

    ##Prior ensemble
    Xb = np.zeros((ni, nj, nv, nens))
    for m in range(nens):
        Xb[:, :, :, m] = gen_random_flow(ni, nj, nv, dx, Vbg, -3) + gen_vortex(ni, nj, nv, Vmax, Rmw, loc_sprd)

    ###Observation
    Yo = np.zeros((nobs*nv))
    Ymask = np.zeros((nobs*nv))
    Yloc = np.zeros((3, nobs*nv))
    Xo = Xt.copy()
    for k in range(nv):
        Xo[:, :, k] += obs_err_std * random_field(ni, obs_err_power_law)
    Yloc = gen_obs_loc(ni, nj, nv, nobs)
    Yo = obs_interp2d(Xo, Yloc)
    Ydist = get_dist(ni, nj, Yloc[0, :], Yloc[1, :], true_center[0], true_center[1])
    Ymask[np.where(Ydist<=obs_range)] = 1
    Yo[np.where(Ymask==0)] = 0.0

    np.save(outdir+dirname+'/Yo.npy', Yo)
    np.save(outdir+dirname+'/Yloc.npy', Yloc)
    np.save(outdir+dirname+'/Yloc.npy', Yloc)

    ##Run filter
    Xa = filter_update(Xb, Yo, Ymask, Yloc, 'EnSRF', obs_err_std*np.ones(1), 0.0*np.ones(1), get_krange(1), (1,), run_alignment=False)
    err = diagnose(Xa, Xt)
    np.save(outdir+dirname+'/EnSRF_full_update.npy', err)

    # for s in range(3):
    #     Xa = filter_update(Xb, Yo, Ymask, Yloc, 'EnSRF', obs_err_std*np.ones(1), 0.0*np.ones(1), get_krange(1), (1,), run_alignment=False)
    #     err = diagnose(Xa, Xt)
    #     np.save(outdir+dirname+'/EnSRF_update_scale{}.npy'.format(, err)


