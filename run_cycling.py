#!/usr/bin/env python
import numpy as np
import sys
from rankine_vortex import *
from data_assimilation import *
from obs_def import *
from multiscale import *
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
    true_size[t] = vortex_size(Xt[:, :, :, t])


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
Yo[np.where(Ymask==0)] = 0.0


##save data
np.save(outdir+'Xt.npy', Xt)
np.save(outdir+'truth_center.npy', true_center)
np.save(outdir+'truth_intensity.npy', true_intensity)
np.save(outdir+'truth_size.npy', true_size)
np.save(outdir+'Yo.npy', Yo)
np.save(outdir+'Yloc.npy', Yloc)
np.save(outdir+'Ymask.npy', Ymask)

Xt = np.load(outdir+'Xt.npy')
true_center = np.load(outdir+'truth_center.npy')
true_intensity = np.load(outdir+'truth_intensity.npy')
true_size = np.load(outdir+'truth_size.npy')
Yo = np.load(outdir+'Yo.npy')
Yloc = np.load(outdir+'Yloc.npy')
Ymask = np.load(outdir+'Ymask.npy')

seed = int(sys.argv[1])  ##random seed
filter_kind = sys.argv[2] #'NoDA'
ns = int(sys.argv[3])    ##number of scales

krange = (1,) # get_krange(ns)
obs_err_infl = np.ones(ns)
local_cutoff = 30*np.ones(ns)
run_alignment = False

casename = filter_kind+'_s{}_{:05d}'.format(ns, seed)

np.random.seed(seed)

X = np.zeros((ni, nj, nv, nens, 2, nt+1))
ens_center = np.zeros((2, nens+1, 2, nt))    ###storm location 0:x 1:y; 0:prior 1:posterior
ens_intensity = np.zeros((nens+1, 2, nt))   ###vortex intensity from ensemble
ens_size = np.zeros((nens+1, 2, nt))   ###vortex size

for m in range(nens):
    X[:, :, :, m, 0, 0] = gen_random_flow(ni, nj, nv, dx, Vbg, -3) + gen_vortex(ni, nj, nv, Vmax, Rmw, loc_sprd)

##start cycling
for t in range(nt):
    ###analysis cycle
    if t>0 and t%obs_t_intv==0 and filter_kind!='NoDA':
        print('running '+filter_kind+' for t={}'.format(t))
        X[:, :, :, :, 1, t] = filter_update(X[:, :, :, :, 0, t], Yo[:, t], Ymask[:, t], Yloc[:, :, t],
                                         filter_kind, obs_err_std*obs_err_infl, local_cutoff,
                                         krange, run_alignment)
    else:
        X[:, :, :, :, 1, t] = X[:, :, :, :, 0, t]

    ###run model forecast
    print('running forecast t={}'.format(t))
    X[:, :, :, :, 0, t+1] = advance_time(X[:, :, :, :, 1, t], dx, dt, smalldt, gen, diss)

    ##diagnose
    for i in range(2):
        for m in range(nens):
            ens_center[:, m, i, t] = vortex_center(X[:, :, :, m, i, t])
            ens_intensity[m, i, t] = vortex_intensity(X[:, :, :, m, i, t])
            ens_size[m, i, t] = vortex_size(X[:, :, :, m, i, t])
        ens_center[:, nens, i, t] = vortex_center(np.mean(X[:, :, :, :, i, t], axis=3))
        ens_intensity[nens, i, t] = vortex_intensity(np.mean(X[:, :, :, :, i, t], axis=3))
        ens_size[nens, i, t] = vortex_size(np.mean(X[:, :, :, :, i, t], axis=3))
        state_err[]

    np.save(outdir+'cycle/'+casename+'_X.npy', X)
    np.save(outdir+'cycle/'+casename+'_center.npy', ens_center)
    np.save(outdir+'cycle/'+casename+'_intensity.npy', ens_intensity)
    np.save(outdir+'cycle/'+casename+'_size.npy', ens_size)
