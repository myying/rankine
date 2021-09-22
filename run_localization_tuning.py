import numpy as np
import sys
import os
from rankine_vortex import *
from obs_def import *
from data_assimilation import *
from multiscale import *
from config import *

realize = int(sys.argv[1])

network_type = 1
loc_sprd = int(sys.argv[2])
ns = int(sys.argv[3])
update_s = 0
local_cutoff_try = (8, 12, 16, 20, 24, 28, 32, 40, 48, 64)
local_dampen_try = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1)

np.random.seed(realize)
dirname = 'localization_tuning/{:04d}/type{}'.format(realize, network_type)
if not os.path.exists(outdir+dirname):
    os.makedirs(outdir+dirname)

##truth
Xt = gen_random_flow(ni, nj, nv, dx, Vbg, -3) + gen_vortex(ni, nj, nv, Vmax, Rmw)

##Observations
if os.path.isfile(outdir+dirname+'/Yo.npy'):
    Yo = np.load(outdir+dirname+'/Yo.npy')
    Yloc = np.load(outdir+dirname+'/Yloc.npy')
    Ymask = np.load(outdir+dirname+'/Ymask.npy')
else:
    nobs, obs_range = gen_network(network_type)
    Yo = np.zeros((nobs*nv))
    Ymask = np.zeros((nobs*nv))
    Yloc = np.zeros((3, nobs*nv))
    Xo = Xt.copy()
    for k in range(nv):
        Xo[:, :, k] += obs_err_std * random_field(ni, obs_err_power_law)
    Yloc = gen_obs_loc(ni, nj, nv, nobs)
    Yo = obs_interp2d(Xo, Yloc)
    Ydist = get_dist(ni, nj, Yloc[0, :], Yloc[1, :], 0.5*ni, 0.5*nj)
    Ymask[np.where(Ydist<=obs_range)] = 1
    Yo[np.where(Ymask==0)] = 0.0
    np.save(outdir+dirname+'/Yo.npy', Yo)
    np.save(outdir+dirname+'/Yloc.npy', Yloc)
    np.save(outdir+dirname+'/Ymask.npy', Ymask)


scenario = '/Lsprd{}/ns{}_u{}_mso'.format(loc_sprd, ns, update_s)
if not os.path.exists(outdir+dirname+scenario):
    os.makedirs(outdir+dirname+scenario)

##Prior ensemble
Xb = np.zeros((ni, nj, nv, nens))
for m in range(nens):
    Xb[:, :, :, m] = gen_random_flow(ni, nj, nv, dx, Vbg, -3) + gen_vortex(ni, nj, nv, Vmax, Rmw, loc_sprd)

rmse = np.zeros((len(local_cutoff_try), len(local_dampen_try)))
rmse[:, :] = np.nan
sprd = np.zeros((len(local_cutoff_try), len(local_dampen_try)))
for i in range(len(local_cutoff_try)):
    for j in range(len(local_dampen_try)):
        print(local_cutoff_try[i], local_dampen_try[j])
        X = filter_update(Xb, Yo, Ymask, Yloc, 'EnSRF', obs_err_std*np.ones(ns),
                          local_cutoff_try[i]*np.ones(ns), local_dampen_try[j]*np.ones(ns),
                          get_krange(ns), get_krange(ns), run_alignment=False, print_out=False, update_scale=update_s, obs_scale=0)
        rmse[i, j] = mean_rmse(X, Xt)
        sprd[i, j] = ens_sprd(X)
        np.save(outdir+dirname+scenario+'/rmse.npy', rmse)
        np.save(outdir+dirname+scenario+'/sprd.npy', sprd)
