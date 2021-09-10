#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from rankine_vortex import *
from obs_def import *
from data_assimilation import *
from multiscale import *
from config import *

np.random.seed(0)

##truth
Xt = gen_vortex(ni, nj, nv, Vmax, Rmw, 0)

##observation
Ymask = np.ones(2)
Yloc = np.zeros((3, 2))

##wind obs at random location
obs_err = 3
obs_th = 45
obs_r = 5
Yloc[0, :] = 0.5*ni + obs_r*np.sin(obs_th*np.pi/180*np.ones(2))
Yloc[1, :] = 0.5*nj + obs_r*np.cos(obs_th*np.pi/180*np.ones(2))
Yloc[2, :] = np.array([0, 1])
Yo = obs_interp2d(Xt, Yloc) + obs_err * np.random.normal(0, 1, 2)

##position obs
# obs_err = 0.5
# Yloc[2, :] = np.array([-1, -1])
# Yo = vortex_center(Xt) + obs_err * np.random.normal(0, 1, 2)

##output state location
out_th = 160
out_r = 6
out_i = 0.5*ni + out_r*np.sin(out_th*np.pi/180)
out_j = 0.5*nj + out_r*np.cos(out_th*np.pi/180)


##prior ensemble
nens = 200
loc_sprd = 0.6*Rmw
loc_bias = 0.0*Rmw
Xb = np.zeros((ni, nj, nv, nens))
for m in range(nens):
    Xb[:, :, :, m] = gen_vortex(ni, nj, nv, Vmax, Rmw, loc_sprd, loc_bias)

filter_kind = ('NoDA', 'EnSRF', 'EnSRF', 'EnSRF', 'EnSRF', 'PF')
ns = (1, 1, 3, 5, 7, 1)
nc = len(filter_kind)
X = np.zeros((ni, nj, nv, nens, nc))
for c in range(nc):
    print(filter_kind[c], 'ns={}'.format(ns[c]))
    X[:, :, :, :, c] = filter_update(Xb, Yo, Ymask, Yloc, filter_kind[c],
                                     obs_err*np.ones(ns[c]), np.zeros(ns[c]),
                                     get_krange(ns[c]), (1,), run_alignment=True)

plt.switch_backend('Agg')
plt.figure(figsize=(12, 10))
ii, jj = np.mgrid[0:ni, 0:nj]
iout = (ii-ni/2)*dx/1000
jout = (jj-nj/2)*dx/1000
cmap = [plt.cm.jet(m) for m in np.linspace(0.2, 0.8, nens)]

for c in range(nc):
    ax = plt.subplot(2, 3, c+1)
    for m in range(0, nens, int(nens/20)):
        wspd = np.sqrt(X[:, :, 0, m, c]**2+X[:, :, 1, m, c]**2)
        ax.contour(iout, jout, wspd, (20,), colors=[cmap[m][0:3]], linewidths=2)
    # Xm = np.mean(X[:, :, :, :, c], axis=3)
    # wspd = np.sqrt(Xm[:, :, 0]**2+Xm[:, :, 1]**2)
    # ax.contour(iout, jout, wspd, (20,), colors='r', linewidths=3)
    wspd = np.sqrt(Xt[:, :, 0]**2+Xt[:, :, 1]**2)
    ax.contour(iout, jout, wspd, (20,), colors='k', linewidths=3)
    ax.plot((Yloc[0, ::2]-ni/2)*dx/1000, (Yloc[1, ::2]-nj/2)*dx/1000, 'k+', markersize=10, markeredgewidth=2)
    ax.plot((out_i-ni/2)*dx/1000, (out_j-nj/2)*dx/1000, 'kx', markersize=7, markeredgewidth=2)
    ax.set_aspect('equal', 'box')
    ax.set_xlim(-135, 135)
    ax.set_ylim(-135, 135)
    ax.set_xticks(np.arange(-120, 121, 60))
    ax.set_yticks(np.arange(-120, 121, 60))
    ax.tick_params(labelsize=12)
    metric = np.zeros(nens)
    for m in range(nens):
        metric[m] = vortex_intensity(X[:, :, :, m, c])
    print(c, np.mean(metric))

plt.savefig('1.pdf')
