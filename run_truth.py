#!/usr/bin/env python
import numpy as np
from rankine_vortex import *
from data_assimilation import *

outdir = '/Users/yueng/scratch/rankine/cycle/'

##model state parameters
ni = 128    # number of grid points i, j directions
nj = 128
nv = 2     # number of variables, (u, v)
dx = 9000
dt = 3600
nt = 12

smalldt = 60
gen = 5e-5
diss = 3e3

### vortex parameters
Vbg = 3   ##background flow amplitude
Vmax = 35     # maximum wind speed (vortex intensity)
Rmw = 5        # radius of maximum wind (vortex size)

##ensemble parameters
nens = 20    ##ensemble size
loc_sprd = 10  ##position spread in prior ensemble

##obs network parameters
nobs = 5000    ##number of observations in entire domain
obs_range = 30  ##radius from vortex center where obs are available (will be assimilated)
obs_err_std = 3.0   ##measurement error
obs_err_power_law = 1
obs_t_intv = 3


##initialize ensemble members and truth
Xt = np.zeros((ni, nj, nv, nt+1))
Xb = np.zeros((ni, nj, nv, nens, nt+1))
Xt[:, :, :, 0] = gen_random_flow(ni, nj, nv, dx, Vbg, -3) + gen_vortex(ni, nj, nv, Vmax, Rmw, 0)
for m in range(nens):
    Xb[:, :, :, m, 0] = gen_random_flow(ni, nj, nv, dx, Vbg, -3) + gen_vortex(ni, nj, nv, Vmax, Rmw, loc_sprd)

##run truth simulation
for t in range(nt):
    print(t)
    Xt[:, :, :, t+1] = advance_time(Xt[:, :, :, t], dx, dt, smalldt, gen, diss)

##get true vortex features
true_center = np.zeros((2, nt))
true_intensity = np.zeros(nt)
true_size = np.zeros(nt)
for t in range(nt):
    true_center[:, t] = vortex_center(Xt[:, :, :, t])
    true_intensity[t] = vortex_intensity(Xt[:, :, :, t])
    true_size[t] = vortex_size(Xt[:, :, :, t], true_center[:, t])


##generate obs
Yo = np.zeros((nobs*nv, nt))
Ymask = np.zeros((nobs*nv, nt))
Yloc = np.zeros((3, nobs*nv, nt))
for t in range(nt):
    ###first add random obs error to truth
    Xo = Xt[:, :, :, t].copy()
    for k in range(nv):
        Xo[:, :, k] += obs_err_std * random_field(ni, obs_err_power_law)
    ###interpolate to obs network
    Yloc[:, :, t] = gen_obs_loc(ni, nj, nv, nobs)
    Yo[:, t] = obs_interp2d(Xo, Yloc[:, :, t])
    Ydist = get_dist(ni, nj, Yloc[0, :, t], Yloc[1, :, t], true_center[0, t], true_center[1, t])
    Ymask[np.where(Ydist<=obs_range), t] = 1
    Yo[np.where(Ymask==0), t] = 0.0


##save data
np.save(outdir+'Xt.npy', Xt)
np.save(outdir+'truth_center.npy', true_center)
np.save(outdir+'truth_intensity.npy', true_intensity)
np.save(outdir+'truth_size.npy', true_size)
np.save(outdir+'Yo.npy', Yo)
np.save(outdir+'Yloc.npy', Yloc)
np.save(outdir+'Ymask.npy', Ymask)
