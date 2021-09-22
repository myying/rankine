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

nens = 20 # ensemble size
loc_bias = 0
loc_sprd = int(sys.argv[2])
ns = int(sys.argv[3])

network_type = 1
nobs, obs_range = gen_network(network_type)

np.random.seed(realize)
dirname = 'full_network/type{}/{:04d}'.format(network_type, realize)
if not os.path.exists(outdir+dirname):
    os.makedirs(outdir+dirname)

###truth and observation
if os.path.isfile(outdir+dirname+'/Xt.npy'):
    Xt = np.load(outdir+dirname+'/Xt.npy')
    Yo = np.load(outdir+dirname+'/Yo.npy')
    Yloc = np.load(outdir+dirname+'/Yloc.npy')
    Ymask = np.load(outdir+dirname+'/Ymask.npy')
else:
    Xt = gen_random_flow(ni, nj, nv, dx, Vbg, -3) + gen_vortex(ni, nj, nv, Vmax, Rmw)
    true_center = vortex_center(Xt)
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
    np.save(outdir+dirname+'/Xt.npy', Xt)
    np.save(outdir+dirname+'/Yo.npy', Yo)
    np.save(outdir+dirname+'/Yloc.npy', Yloc)
    np.save(outdir+dirname+'/Ymask.npy', Ymask)

scenario = "/Lbias{}/Lsprd{}/N{}".format(loc_bias, loc_sprd, nens)
if not os.path.exists(outdir+dirname+scenario):
    os.makedirs(outdir+dirname+scenario)

##Prior ensemble
Xb = np.zeros((ni, nj, nv, nens))
for m in range(nens):
    Xb[:, :, :, m] = gen_random_flow(ni, nj, nv, dx, Vbg, -3) + gen_vortex(ni, nj, nv, Vmax, Rmw, loc_sprd, loc_bias)
if not os.path.isfile(outdir+dirname+scenario+'/NoDA.npy'):
    err = diagnose(Xb, Xt)
    np.save(outdir+dirname+scenario+'/NoDA.npy', err)

##Run filter with MSA:
if not os.path.isfile(outdir+dirname+scenario+'/EnSRF_s{}.npy'.format(ns)):
    Xa = filter_update(Xb, Yo, Ymask, Yloc, 'EnSRF', obs_err_std*np.ones(ns),
                        get_local_cutoff(ns), get_local_dampen(ns), get_krange(ns), get_krange(1), run_alignment=True)
    err = diagnose(Xa, Xt)
    np.save(outdir+dirname+scenario+'/EnSRF_s{}.npy'.format(ns), err)
##mso
if ns>1 and not os.path.isfile(outdir+dirname+scenario+'/EnSRF_s{}_mso.npy'.format(ns)):
    Xa = filter_update(Xb, Yo, Ymask, Yloc, 'EnSRF', obs_err_std*np.ones(ns),
                        get_local_cutoff(ns), get_local_dampen(ns), get_krange(ns), get_krange(ns), run_alignment=True)
    err = diagnose(Xa, Xt)
    np.save(outdir+dirname+scenario+'/EnSRF_s{}_mso.npy'.format(ns), err)
