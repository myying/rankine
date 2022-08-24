import numpy as np
import matplotlib.pyplot as plt
from rankine_vortex import *
from data_assimilation import *
from multiscale import *
from config import *

def set_axes(ax):
    ax.set_aspect('equal', 'box')
    ax.set_xlim(14, 114)
    ax.set_ylim(14, 114)
    ax.set_xticks(np.arange(20, 109, 22))
    ax.set_yticks(np.arange(20, 109, 22))
    ax.set_xticklabels(np.arange(-400, 401, 200))
    ax.set_yticklabels(np.arange(-400, 401, 200))
    ax.tick_params(labelsize=12)

np.random.seed(100)

Xt = gen_random_flow(ni, nj, nv, dx, Vbg, -3) + gen_vortex(ni, nj, nv, Vmax, Rmw, 0)
ii, jj = np.mgrid[0:ni, 0:nj]

ns = 4 ##>=4
Xsc = np.zeros((ni, nj, nv, ns))

plt.switch_backend('Agg')
plt.figure(figsize=(18, 18))


# c = ax.contourf(ii, jj, Xt[:, :, 0], np.arange(-40, 41, 4), cmap='bwr')
# c = ax.contourf(ii, jj, get_wspd(Xt), np.arange(0, 41, 2), cmap='bwr')
# ax.set_aspect('equal')
# ax.tick_params(labelsize=13)
# cb = plt.colorbar(c)
# cb.ax.tick_params(labelsize=12)

###power spectrum panel
ax = plt.subplot(4, ns, 1)
wn, pwr = pwr_spec(Xt)
ax.loglog(wn, pwr, color='black', linewidth=3, label='total')
for s in range(ns):
    Xs = get_scale(Xt, get_krange(ns), s)
    wn, pwr = pwr_spec(Xs)
    ax.loglog(wn, pwr, linewidth=2, label='s={}'.format(s+1)) #, color=cmap[s][0:3])
ax.set_xlim([1, 40])
ax.set_ylim([1e-3, 1e2])
ax.legend(fontsize=12)
ax.tick_params(labelsize=12)

##full state
ax = plt.subplot(4, ns, 2)
Xout = Xt.copy()
Xout[np.where(Xout>30)] = 30
Xout[np.where(Xout<-30)] = -30
c = ax.contourf(ii, jj, Xout[:, :, 0], np.arange(-30, 31, 2), cmap='bwr')
plt.colorbar(c)
set_axes(ax)
for s in range(ns):
    #state SCs
    ax = plt.subplot(4, ns, ns+s+1)
    Xs = get_scale(Xt, get_krange(ns), s)
    ax.contourf(ii, jj, Xs[:, :, 0], np.arange(-30, 31, 2), cmap='bwr')
    set_axes(ax)

for network_type in (1, 2):
    np.random.seed(101)
    nobs, obs_range = gen_network(network_type)
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

    ##full obs
    ax = plt.subplot(4, ns, network_type+2)
    plot_obs(ax, ni, nj, nv, Yo, Ymask, Yloc)
    set_axes(ax)

    ###state SCs
    for s in range(ns):
        ##obs SCs
        ax = plt.subplot(4, ns, (network_type+1)*ns+s+1)
        Yos = obs_get_scale(ni, nj, nv, Yo, Ymask, Yloc, get_krange(ns), s)
        plot_obs(ax, ni, nj, nv, Yos, Ymask, Yloc)
        set_axes(ax)

plt.savefig('out.pdf')
