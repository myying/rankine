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
bkg_phase_err = float(sys.argv[3])

network_type = 1
nobs, obs_range = gen_network(network_type)

np.random.seed(realize)
dirname = 'full_network/type{}/{:04d}'.format(network_type, realize)
if not os.path.exists(outdir+dirname):
    os.makedirs(outdir+dirname)

###truth and observation
if os.path.isfile(outdir+dirname+'/Xt.npy'):
    bkg_flow = np.load(outdir+dirname+'/bkg_flow.npy')
    vortex = np.load(outdir+dirname+'/vortex.npy')
    Xt = np.load(outdir+dirname+'/Xt.npy')
    Yo = np.load(outdir+dirname+'/Yo.npy')
    Yloc = np.load(outdir+dirname+'/Yloc.npy')
    Ymask = np.load(outdir+dirname+'/Ymask.npy')
else:
    bkg_flow = gen_random_flow(ni, nj, nv, dx, Vbg, -3)
    vortex = gen_vortex(ni, nj, nv, Vmax, Rmw)
    Xt = bkg_flow + vortex
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
    np.save(outdir+dirname+'/bkg_flow.npy', bkg_flow)
    np.save(outdir+dirname+'/vortex.npy', vortex)
    np.save(outdir+dirname+'/Yo.npy', Yo)
    np.save(outdir+dirname+'/Yloc.npy', Yloc)
    np.save(outdir+dirname+'/Ymask.npy', Ymask)

scenario = "/Lbias{}/Lsprd{}/phase{}/N{}".format(loc_bias, loc_sprd, bkg_phase_err, nens)
if not os.path.exists(outdir+dirname+scenario):
    os.makedirs(outdir+dirname+scenario)

##Prior ensemble
bkg_flow_ens = np.zeros((ni, nj, nv, nens))
vortex_ens = np.zeros((ni, nj, nv, nens))
u = np.zeros((ni, nj, nv, nens))
v = np.zeros((ni, nj, nv, nens))
for m in range(nens):
    u[:, :, :, m] = np.random.normal(0, loc_sprd)
    v[:, :, :, m] = np.random.normal(0, loc_sprd)
    vortex_ens[:, :, :, m] = vortex
    bkg_flow_ens[:, :, :, m] = bkg_flow
vortex_ens = warp(vortex_ens, -u, -v)
bkg_flow_ens = warp(bkg_flow_ens, -u*bkg_phase_err, -v*bkg_phase_err)
for m in range(nens):
    bkg_flow_ens[:, :, :, m] += gen_random_flow(ni, nj, nv, dx, 0.6*Vbg*(1-bkg_phase_err), -3)
Xb = bkg_flow_ens + vortex_ens

if not os.path.isfile(outdir+dirname+scenario+'/NoDA.npy'):
    err = diagnose(Xb, Xt)
    np.save(outdir+dirname+scenario+'/NoDA.npy', err)

##Run filter with MSA:
for ns in (1, 2, 3, 4, 5, 6, 7):
    if not os.path.isfile(outdir+dirname+scenario+'/EnSRF_s{}.npy'.format(ns)):
        Xa = filter_update(Xb, Yo, Ymask, Yloc, 'EnSRF', obs_err_std*np.ones(ns),
                            get_local_cutoff(ns), get_local_dampen(ns), get_krange(ns), get_krange(1), run_alignment=True)
        err = diagnose(Xa, Xt)
        np.save(outdir+dirname+scenario+'/EnSRF_s{}.npy'.format(ns), err)

##add mso cases, krange_obs = get_krange(ns)
for ns in (2, 3, 4):
    if not os.path.isfile(outdir+dirname+scenario+'/EnSRF_s{}_mso.npy'.format(ns)):
        Xa = filter_update(Xb, Yo, Ymask, Yloc, 'EnSRF', obs_err_std*np.ones(ns),
                            get_local_cutoff(ns), get_local_dampen(ns), get_krange(ns), get_krange(ns), run_alignment=True)
        err = diagnose(Xa, Xt)
        np.save(outdir+dirname+scenario+'/EnSRF_s{}_mso.npy'.format(ns), err)
