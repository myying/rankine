import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from rankine_vortex import *
from obs_def import *
from data_assimilation import *
from multiscale import *
from config import *

realize = 1 #int(sys.argv[1])

network_type = 2
loc_sprd = 3 #int(sys.argv[2])
ns = 5 #int(sys.argv[3])
update_s = 0
run_alignment = False
local_cutoff = 40
local_dampen = 0.9

np.random.seed(realize)

##truth
Xt = gen_random_flow(ni, nj, nv, dx, Vbg, -3) + gen_vortex(ni, nj, nv, Vmax, Rmw)

##Observations
if network_type==1:  ##global network
    nobs = 1000
    obs_range = 200
if network_type==2:  ##targeted network
    nobs = 6000
    obs_range = 30
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

##Prior ensemble
Xb = np.zeros((ni, nj, nv, nens))
for m in range(nens):
    Xb[:, :, :, m] = gen_random_flow(ni, nj, nv, dx, Vbg, -3) + gen_vortex(ni, nj, nv, Vmax, Rmw, loc_sprd)

X = filter_update(Xb, Yo, Ymask, Yloc, 'EnSRF', obs_err_std*np.ones(ns),
                    local_cutoff*np.ones(ns), local_dampen*np.ones(ns),
                    get_krange(ns), (16,), run_alignment, update_scale=update_s)

ii, jj = np.mgrid[0:ni, 0:nj]
plt.figure(figsize=(8,8))
ax = plt.subplot(221)
ax.contourf(ii, jj, Xt[:, :, 0], np.arange(-30, 30, 2), cmap='bwr')
ax = plt.subplot(222)
plot_obs(ax, ni, nj, nv, Yo, Ymask, Yloc)
ax = plt.subplot(223)
ax.contourf(ii, jj, np.mean(Xb, axis=3)[:, :, 0], np.arange(-30, 30, 2), cmap='bwr')
ax = plt.subplot(224)
ax.contourf(ii, jj, np.mean(X, axis=3)[:, :, 0], np.arange(-30, 30, 2), cmap='bwr')
plt.show()
