import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from rankine_vortex import *
from obs_def import *
from data_assimilation import *
from multiscale import *
from config import *

realize = 10 #int(sys.argv[1])
ns = int(sys.argv[1])
ns_obs = int(sys.argv[2])

loc_sprd = 5
vmax_sprd = 0
size_sprd = 0.0
bkg_phase_err = 1.0

nens = 20
nobs = 500
obs_range = 100
krange = get_krange(ns)
krange_obs = get_krange(ns_obs)

np.random.seed(realize)
bkg_flow = gen_random_flow(ni, nj, nv, dx, Vbg, -3)
vortex = gen_vortex(ni, nj, nv, Vmax, Rmw)

##truth
Xt = bkg_flow + vortex

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
bkg_flow_ens = np.zeros((ni, nj, nv, nens))
vortex_ens = np.zeros((ni, nj, nv, nens))
u = np.zeros((ni, nj, nv, nens))
v = np.zeros((ni, nj, nv, nens))
for m in range(nens):
    u[:, :, :, m] = np.random.normal(0, loc_sprd)
    v[:, :, :, m] = np.random.normal(0, loc_sprd)
    Vmax_pert = np.maximum(Vmax + np.random.normal(0, vmax_sprd), 20)
    Rmw_pert = np.maximum(Rmw + np.random.normal(0, size_sprd), 3)
    vortex_ens[:, :, :, m] = gen_vortex(ni, nj, nv, Vmax_pert, Rmw_pert)
    bkg_flow_ens[:, :, :, m] = bkg_flow
vortex_ens = warp(vortex_ens, -u, -v)
bkg_flow_ens = warp(bkg_flow_ens, -u*bkg_phase_err, -v*bkg_phase_err)
for m in range(nens):
    bkg_flow_ens[:, :, :, m] += gen_random_flow(ni, nj, nv, dx, 0.6*Vbg*(1-bkg_phase_err), -3)
Xb = bkg_flow_ens + vortex_ens

X = Xb.copy()
X = filter_update(X, Yo, Ymask, Yloc, 'EnSRF', obs_err_std*np.ones(ns), get_local_cutoff(ns), get_local_dampen(ns), krange, krange_obs, run_alignment=True, update_scale=-1)
err = diagnose(X, Xt)
print('domain-avg err=', err[nens, 0])
print('pos err=', np.mean(err[0:nens, 1]))
print('intensity err=', np.mean(err[0:nens, 2]))
print('size err=', np.mean(err[0:nens, 3]))

m = 0
ii, jj = np.mgrid[0:ni, 0:nj]
plt.figure(figsize=(8,11))
ax = plt.subplot(321)
ax.contourf(ii, jj, Xt[:, :, 0], np.arange(-40, 40, 2), cmap='bwr')
ax.set_title('truth')
ax = plt.subplot(322)
plot_obs(ax, ni, nj, nv, Yo, Ymask, Yloc)
ax.set_title('obs')
ax = plt.subplot(323)
ax.contourf(ii, jj, np.mean(Xb, axis=3)[:, :, 0], np.arange(-40, 40, 2), cmap='bwr')
ax.set_title('prior mean')
# ax.contourf(ii, jj, Xb[:, :, 0, m], np.arange(-30, 30, 2), cmap='bwr')
ax = plt.subplot(324)
ax.contourf(ii, jj, np.mean(X, axis=3)[:, :, 0], np.arange(-40, 40, 2), cmap='bwr')
ax.set_title('posterior mean')
# ax.contourf(ii, jj, X[:, :, 0, m], np.arange(-30, 30, 2), cmap='bwr')

ii, jj = np.mgrid[0:ni, 0:nj]
iout = (ii-ni/2)*dx/1000
jout = (jj-nj/2)*dx/1000
cmap = [plt.cm.jet(m) for m in np.linspace(0.2, 0.8, nens)]
ax = plt.subplot(325)
for m in range(nens):
    wspd = np.sqrt(Xb[:, :, 0, m]**2+Xb[:, :, 1, m]**2)
    ax.contour(iout, jout, wspd, (20,), colors=[cmap[m][0:3]], linewidths=2)
wspd = np.sqrt(Xt[:, :, 0]**2+Xt[:, :, 1]**2)
ax.contour(iout, jout, wspd, (20,), colors='k', linewidths=3)
ax.set_aspect('equal', 'box')
ax.set_xlim(-135, 135)
ax.set_ylim(-135, 135)
ax.set_xticks(np.arange(-120, 121, 60))
ax.set_yticks(np.arange(-120, 121, 60))
ax.tick_params(labelsize=12)
ax = plt.subplot(326)
for m in range(nens):
    wspd = np.sqrt(X[:, :, 0, m]**2+X[:, :, 1, m]**2)
    ax.contour(iout, jout, wspd, (20,), colors=[cmap[m][0:3]], linewidths=2)
wspd = np.sqrt(Xt[:, :, 0]**2+Xt[:, :, 1]**2)
ax.contour(iout, jout, wspd, (20,), colors='k', linewidths=3)
ax.set_aspect('equal', 'box')
ax.set_xlim(-135, 135)
ax.set_ylim(-135, 135)
ax.set_xticks(np.arange(-120, 121, 60))
ax.set_yticks(np.arange(-120, 121, 60))
ax.tick_params(labelsize=12)

plt.show()

