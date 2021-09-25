#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from rankine_vortex import *
from obs_def import *
from data_assimilation import *
from multiscale import *
from config import *

realize = int(sys.argv[1])
filter_kind = sys.argv[2]  ##NoDA or EnSRF
ns = int(sys.argv[3])    ##number of scales

nens = get_equal_cost_nens(20, ns)
loc_sprd = 3
bkg_phase_err = 1.0
bkg_amp_err = 0.0
network_type = 2
model_kind = 'perfect_model'

nobs, obs_range = gen_network(network_type)
if model_kind == 'perfect_model':
    gen_ens = gen*np.ones(nens)
if model_kind == 'imperfect_model':
    gen_ens = np.random.uniform(2, 6, nens)*1e-5

np.random.seed(realize)
dirname = 'cycling/{}/type{}/{:04d}'.format(model_kind, network_type, realize)
if not os.path.exists(outdir+dirname):
    os.makedirs(outdir+dirname)

##truth and observations
if os.path.isfile(outdir+dirname+'/Xt.npy'):
    bkg_flow = np.load(outdir+dirname+'/bkg_flow.npy')
    vortex = np.load(outdir+dirname+'/vortex.npy')
    Xt = np.load(outdir+dirname+'/Xt.npy')
    true_center = np.load(outdir+dirname+'/true_center.npy')
    true_intensity = np.load(outdir+dirname+'/true_intensity.npy')
    true_size = np.load(outdir+dirname+'/true_size.npy')
    Yo = np.load(outdir+dirname+'/Yo.npy')
    Yloc = np.load(outdir+dirname+'/Yloc.npy')
    Ymask = np.load(outdir+dirname+'/Ymask.npy')
else:
    ##generate truth
    Xt = np.zeros((ni, nj, nv, nt+1))
    bkg_flow = gen_random_flow(ni, nj, nv, dx, Vbg, -3)
    vortex = gen_vortex(ni, nj, nv, Vmax, Rmw)
    Xt[:, :, :, 0] = bkg_flow + vortex
    for t in range(nt):
        Xt[:, :, :, t+1] = advance_time(Xt[:, :, :, t], dx, dt, smalldt, gen, diss)
    true_center = np.zeros((2, nt))
    true_intensity = np.zeros(nt)
    true_size = np.zeros(nt)
    for t in range(nt):
        true_center[:, t] = vortex_center(Xt[:, :, :, t])
        true_intensity[t] = vortex_intensity(Xt[:, :, :, t])
        true_size[t] = vortex_size(Xt[:, :, :, t])
    ##generate observations
    Yo = np.zeros((nobs*nv, nt))
    Ymask = np.zeros((nobs*nv, nt))
    Yloc = np.zeros((3, nobs*nv, nt))
    for t in range(nt):
        Xo = Xt[:, :, :, t].copy()
        for k in range(nv):
            Xo[:, :, k] += obs_err_std * random_field(ni, obs_err_power_law)
        Yloc[:, :, t] = gen_obs_loc(ni, nj, nv, nobs)
        Yo[:, t] = obs_interp2d(Xo, Yloc[:, :, t])
        Ydist = get_dist(ni, nj, Yloc[0, :, t], Yloc[1, :, t], true_center[0, t], true_center[1, t])
        Ymask[np.where(Ydist<=obs_range), t] = 1
        Yo[np.where(Ymask[:, t]==0), t] = 0.0
    ##save files
    np.save(outdir+dirname+'/bkg_flow.npy', bkg_flow)
    np.save(outdir+dirname+'/vortex.npy', vortex)
    np.save(outdir+dirname+'/Xt.npy', Xt)
    np.save(outdir+dirname+'/true_center.npy', true_center)
    np.save(outdir+dirname+'/true_intensity.npy', true_intensity)
    np.save(outdir+dirname+'/true_size.npy', true_size)
    np.save(outdir+dirname+'/Yo.npy', Yo)
    np.save(outdir+dirname+'/Yloc.npy', Yloc)
    np.save(outdir+dirname+'/Ymask.npy', Ymask)

scenario = "/Lsprd{}/phase{}_amp{}".format(loc_sprd, bkg_phase_err, bkg_amp_err)
if not os.path.exists(outdir+dirname+scenario):
    os.makedirs(outdir+dirname+scenario)

##Prior ensemble
np.random.seed(realize)
bkg_flow_ens = np.zeros((ni, nj, nv, nens))
vortex_ens = np.zeros((ni, nj, nv, nens))
u = np.zeros((ni, nj, nv, nens))
v = np.zeros((ni, nj, nv, nens))
for m in range(nens):
    u[:, :, :, m] = np.random.normal(0, loc_sprd)
    v[:, :, :, m] = np.random.normal(0, loc_sprd)
    vortex_ens[:, :, :, m] = vortex
    bkg_flow_ens[:, :, :, m] = bkg_flow
vortex_ens = warp(vortex_ens, -u, -v)
bkg_flow_ens = warp(bkg_flow_ens, -u*bkg_phase_err, -v*bkg_phase_err)
for m in range(nens):
    bkg_flow_ens[:, :, :, m] += gen_random_flow(ni, nj, nv, dx, 0.6*Vbg*bkg_amp_err, -3)
X = bkg_flow_ens + vortex_ens

if not os.path.isfile(outdir+dirname+scenario+'/{}_s{}.npy'.format(filter_kind, ns)):
    err = np.zeros((nens+1, 4, 2, nt))

    ##start cycling
    for t in range(nt):
        # print(t)
        ##diagnose prior
        err[:, :, 0, t] = diagnose(X, Xt[:, :, :, t])
        ##run filter update
        if filter_kind=='EnSRF' and t<10 and t%obs_t_intv==0:
            X = filter_update(X, Yo[:, t], Ymask[:, t], Yloc[:, :, t], 'EnSRF', obs_err_std*np.ones(ns),
                              get_local_cutoff(ns), get_local_dampen(ns), get_krange(ns), get_krange(1), run_alignment=True)
        ##diagnose posterior
        err[:, :, 1, t] = diagnose(X, Xt[:, :, :, t])

        # plt.figure(figsize=(12, 10))
        # ax = plt.subplot(111)
        # ii, jj = np.mgrid[0:ni, 0:nj]
        # cmap = [plt.cm.jet(m) for m in np.linspace(0.2, 0.8, nens)]
        # for m in range(0, nens, int(nens/20)):
        #     wspd = np.sqrt(X[:, :, 0, m]**2+X[:, :, 1, m]**2)
        #     ax.contour(ii, jj, wspd, (20,), colors=[cmap[m][0:3]], linewidths=2)
        # wspd = np.sqrt(Xt[:, :, 0, t]**2+Xt[:, :, 1, t]**2)
        # ax.contour(ii, jj, wspd, (20,), colors='k', linewidths=3)
        # ax.set_aspect('equal', 'box')
        # plt.show()

        ##run forecast
        X = advance_time(X, dx, dt, smalldt, gen_ens, diss)
    ##save diagnose file
    np.save(outdir+dirname+scenario+'/{}_s{}.npy'.format(filter_kind, ns), err)

