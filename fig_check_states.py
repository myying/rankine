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

nobs = 500
obs_range = 100
loc_sprd = 5 #int(sys.argv[1])
ns = int(sys.argv[1])
krange = get_krange(ns)
krange_obs = get_krange(ns)

np.random.seed(realize)
Vbg = 5
bkg_flow = gen_random_flow(ni, nj, nv, dx, Vbg, -3)

##truth
Xt = bkg_flow + gen_vortex(ni, nj, nv, Vmax, Rmw)

##Observations
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
u = np.zeros((ni, nj, nv, nens))
v = np.zeros((ni, nj, nv, nens))
for m in range(nens):
    # Xb[:, :, :, m] = bkg_flow + gen_random_flow(ni, nj, nv, dx, 0.0*Vbg, -3) + gen_vortex(ni, nj, nv, Vmax, Rmw, loc_sprd)
    u[:, :, :, m] = np.random.normal(0, loc_sprd)
    v[:, :, :, m] = np.random.normal(0, loc_sprd)
    Xb[:, :, :, m] = Xt + gen_random_flow(ni, nj, nv, dx, 0.3*Vbg, -3)
Xb = warp(Xb, -u, -v)

X = Xb.copy()
X = filter_update(X, Yo, Ymask, Yloc, 'EnSRF', obs_err_std*np.ones(ns), get_local_cutoff(ns), get_local_dampen(ns), krange, krange_obs, run_alignment=True, update_scale=-1)
err = diagnose(X, Xt)
print(err[nens, 0], np.mean(err[0:nens, 1]), np.mean(err[0:nens, 2]), np.mean(err[0:nens, 3]))

m = 0
ii, jj = np.mgrid[0:ni, 0:nj]
plt.figure(figsize=(8,8))
ax = plt.subplot(221)
ax.contourf(ii, jj, Xt[:, :, 0], np.arange(-30, 30, 2), cmap='bwr')
ax = plt.subplot(222)
plot_obs(ax, ni, nj, nv, Yo, Ymask, Yloc)
ax = plt.subplot(223)
ax.contourf(ii, jj, np.mean(Xb, axis=3)[:, :, 0], np.arange(-30, 30, 2), cmap='bwr')
# ax.contourf(ii, jj, Xb[:, :, 0, m], np.arange(-30, 30, 2), cmap='bwr')
ax = plt.subplot(224)
ax.contourf(ii, jj, np.mean(X, axis=3)[:, :, 0], np.arange(-30, 30, 2), cmap='bwr')
# ax.contourf(ii, jj, X[:, :, 0, m], np.arange(-30, 30, 2), cmap='bwr')
plt.show()

