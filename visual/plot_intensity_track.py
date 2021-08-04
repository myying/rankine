#!/usr/bin/env python
import numpy as np
import rankine_vortex as rv
import matplotlib.pyplot as plt
import sys

outdir = '/storage/windows10/scratch/rankine/cycle/'
ni = 128  # number of grid points i, j directions
nj = 128
nv = 2   # number of variables, (u, v)
dx = 9000

plt.figure(figsize=(15, 10))
casename = ('NoDA_s1', 'obs.intv.1/EnSRF_s1', 'obs.intv.1/EnSRF_s4') #sys.argv[1] #'EnSRF_s3'
col = ([.3, .6, .3], [.8, .3, .1], [.2, .7, .9])
for c in range(3):
    X = np.load(outdir+'truth_state.npy')
    loc = np.load(outdir+'truth_ij.npy')
    wind = np.load(outdir+'truth_wind.npy')
    Xens = np.load(outdir+casename[c]+'_ens.npy')
    loc_ens = np.load(outdir+casename[c]+'_ij.npy')
    wind_ens = np.load(outdir+casename[c]+'_wind.npy')
    nX, nens, nc, nt = Xens.shape
    tt = np.arange(0, nt*2, 2) ##time in h
    cmap = [plt.cm.jet(x) for x in np.linspace(0, 1,nens)]

    ii, jj = np.mgrid[0:ni, 0:nj]
    u, v = rv.X2uv(ni, nj, X[:, -1])
    zeta = rv.uv2zeta(u, v, dx)

    ##track errors
    ax = plt.subplot(231)
    # c = ax.contourf(ii, jj, zeta, np.arange(-3, 3, 0.1)*1e-3, cmap='bwr')
    # c = ax.contourf(ii, jj, u, np.arange(-70, 70, 5), cmap='bwr')
    for m in range(nens):
        ax.plot(loc_ens[0, m, 1, 0:nt], loc_ens[1, m, 1, 0:nt], color=col[c], marker=None)  ##member
    # ax.plot(loc[0, nens, 0:nt], loc_ens[1, nens, 0:nt], 'g', linewidth=3) ##ens mean
    ax.plot(loc[0, 0:nt], loc[1, 0:nt], 'k', linewidth=3)  ##true
    ax.set_xlim(20, 80)
    ax.set_ylim(40, 100)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Track ensemble', fontsize=20)
    ax.tick_params(labelsize=15)

    ax = plt.subplot(234)
    loc_rmse = 0
    for m in range(nens):
        loc_rmse += (loc[0, 0:nt] - loc_ens[0, m, 1, 0:nt])**2 + (loc[1, 0:nt] - loc_ens[1, m, 1, 0:nt])**2
    loc_rmse = np.sqrt(loc_rmse/nens)
    ax.plot(tt, loc_rmse, color=col[c], linewidth=3)
    # loc_m_rmse = np.sqrt((loc[0, 0:nt] - loc_ens[0, nens, 1, 0:nt])**2 + (loc[1, 0:nt] - loc_ens[1, nens, 1, 0:nt])**2)
    # ax.plot(loc_m_rmse)
    ax.set_ylim(0, 20)
    ax.set_xlabel('hour')
    ax.set_title('Track error', fontsize=20)
    ax.tick_params(labelsize=15)

    ##intensity errors
    ax = plt.subplot(232)
    for m in range(nens):
        ax.plot(tt, wind_ens[m, 1, 0:nt], color=col[c], marker=None)
    ax.plot(tt, wind[0:nt], 'k', linewidth=3)
    ax.set_ylim(10, 60)
    ax.set_xlabel('hour')
    ax.set_title('Vmax ensemble', fontsize=20)
    ax.tick_params(labelsize=15)

    ax = plt.subplot(235)
    wind_rmse = 0
    for m in range(nens):
        wind_rmse += (wind[0:nt] - wind_ens[m, 1, 0:nt])**2
    wind_rmse = np.sqrt(wind_rmse/nens)
    ax.plot(tt, wind_rmse, color=col[c], linewidth=3)
    ax.set_ylim(0, 20)
    ax.set_xlabel('hour')
    ax.set_title('Vmax error', fontsize=20)
    ax.tick_params(labelsize=15)

    ##physics space rmse
    ax = plt.subplot(236)
    rmse = np.zeros(nt)
    for t in range(nt):
        um, vm = rv.X2uv(ni, nj, np.mean(Xens[:, 0:nens, 1, t], axis=1))
        ut, vt = rv.X2uv(ni, nj, X[:, t])
        sq_err = (um - ut)**2 + (vm - vt)**2
        loc_i = int(loc[0, t])
        loc_j = int(loc[1, t])
        buff = 10
        rmse[t] = np.sqrt(np.mean(sq_err[loc_i-buff:loc_i+buff, loc_j-buff:loc_j+buff]))
    ax.plot(tt, rmse, color=col[c], linewidth=3)
    ax.set_ylim(0, 30)
    ax.set_xlabel('hour')
    ax.set_title('domain-avg wind error', fontsize=20)
    ax.tick_params(labelsize=15)

plt.tight_layout()
plt.savefig('1.pdf')
# plt.show()
