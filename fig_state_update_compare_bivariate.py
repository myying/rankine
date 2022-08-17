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
obs_err = 3
obs_th = 45
obs_r = 5
Ymask = np.ones(2)
Yloc = np.zeros((3, 2))
Yloc[0, :] = 0.5*ni + obs_r*np.sin(obs_th*np.pi/180*np.ones(2))
Yloc[1, :] = 0.5*nj + obs_r*np.cos(obs_th*np.pi/180*np.ones(2))
Yloc[2, :] = np.array([0, 1])
Yt = obs_interp2d(Xt, Yloc)
Yo = Yt + obs_err * np.random.normal(0, 1, 2)

##prior ensemble
nens = 200
loc_sprd = 3
Xb = np.zeros((ni, nj, nv, nens))
for m in range(nens):
    Xb[:, :, :, m] = gen_vortex(ni, nj, nv, Vmax, Rmw, loc_sprd)

out_th = 160
out_r = 6
Yloc_out = np.zeros((3, 1))
Yloc_out[0, :] = 0.5*ni + out_r*np.sin(out_th*np.pi/180)
Yloc_out[1, :] = 0.5*nj + out_r*np.cos(out_th*np.pi/180)
Yloc_out[2, :] = np.array([0])
Ytout = obs_interp2d(Xt, Yloc_out)

Y = np.zeros((3, 25, nv, nens))
Yout = np.zeros((3, 25, nens))
Y[:, :, :, :] = np.nan
Yout[:, :, :] = np.nan

c = 0  ##single scale EnSRF, save after updateing each obs
X = Xb.copy()
for p in range(nv+1):
    for m in range(nens):
        Y[c, p, :, m] = obs_interp2d(X[:, :, :, m], Yloc)
        Yout[c, p, m] = obs_interp2d(X[:, :, :, m], Yloc_out)
    if p<nv:
        X = filter_update(X, Yo[p:p+1], Ymask[p:p+1], Yloc[:, p:p+1], 'EnSRF', (obs_err,), (0,), (1,), (16,), (1,), False)

c = 1  ##MSA EnSRF, save after each scale iteration
ns = 5
krange = get_krange(ns)
X = Xb.copy()
for s in range(ns+1):
    for m in range(nens):
        Y[c, s, :, m] = obs_interp2d(X[:, :, :, m], Yloc)
        Yout[c, s, m] = obs_interp2d(X[:, :, :, m], Yloc_out)
    if s<ns:
        X = filter_update(X, Yo, Ymask, Yloc, 'EnSRF', obs_err*np.ones(ns), np.zeros(ns), np.ones(ns), krange, (1,), True, update_scale=s)

c = 2 ##PF solution, just one step
X = Xb.copy()
for j in range(2):
    for m in range(nens):
        Y[c, j, :, m] = obs_interp2d(X[:, :, :, m], Yloc)
        Yout[c, j, m] = obs_interp2d(X[:, :, :, m], Yloc_out)
    if j<1:
        X = filter_update(X, Yo, Ymask, Yloc, 'PF', (obs_err,), (0,), (1,), (16,), (1,), False)

##plot
plt.switch_backend('Agg')
plt.figure(figsize=(12, 6))
niter = (nv, ns, 1)  ##number of steps to show in line segments
for c in range(3):
    for i in range(nv):
        ax = plt.subplot(2, 3, 3*i+c+1)
        ax.plot([-50, 50], [Yt[i], Yt[i]], color=[.7, .7, .7], linewidth=2, zorder=-1)
        ax.plot([Ytout, Ytout], [-50, 50], color=[.7, .7, .7], linewidth=2, zorder=-1)
        ax.plot([-50, 50], [Yo[i], Yo[i]], 'r', linewidth=2, zorder=-1)
        ax.scatter(Yout[c, 0, :], Y[c, 0, i, :], s=10, marker='o', color=[.4, .8, .2], edgecolor=[.3, .7, .3], zorder=0)
        ax.scatter(Yout[c, niter[c], :], Y[c, niter[c], i, :], s=20, marker='^', color=[.2, .4, .9], edgecolor=[.3, .3, .7], zorder=0)
        for m in range(0, nens, 40):
            ax.plot(Yout[c, 0:niter[c]+1, m], Y[c, 0:niter[c]+1, i, m], color='black', linewidth=0.7, zorder=1)
        ax.set_xlim(-10, 40)
        if i == 0:
            ax.set_ylim(-40, 20)
        if i == 1:
            ax.set_ylim(-20, 40)


plt.savefig('out.pdf')
