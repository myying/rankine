#!/usr/bin/env python
import numpy as np
from rankine_vortex import *
from data_assimilation import *
from config import *

##initialize ensemble members and truth
Xt = np.zeros((ni, nj, nv, nt+1))
Xt[:, :, :, 0] = gen_random_flow(ni, nj, nv, dx, Vbg, -3) + gen_vortex(ni, nj, nv, Vmax, Rmw, 0)

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
