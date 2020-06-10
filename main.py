#!/usr/bin/env python
import numpy as np
import rankine_vortex as rv
import graphics as g
import matplotlib.pyplot as plt
import data_assimilation as DA
import sys

filter_kind = sys.argv[1] #'EnSRF'
localize_cutoff = 0

ni = 128  # number of grid points i, j directions
nj = 128
nv = 2   # number of variables, (u, v)
nens = int(sys.argv[2]) # ensemble size

### Rankine Vortex definition, truth
Rmw = 5    # radius of maximum wind
Vmax = 50   # maximum wind speed
Vout = 0    # wind speed outside of vortex
iStorm = 63 # location of vortex in i, j
jStorm = 63

##truth
iX, jX = rv.make_coords(ni, nj)
Xt = rv.make_state(ni, nj, nv, iStorm, jStorm, Rmw, Vmax, Vout)

##DA trials
nrealize = 100
Xa = np.zeros((nrealize, nens, ni*nj*nv))
for realize in range(nrealize):
  np.random.seed(realize)  # fix random number seed, make results predictable

  ##Prior ensemble
  Csprd = int(sys.argv[3])
  iBias = 0
  jBias = 0
  Rsprd = 0
  Vsprd = 0
  Xb = np.zeros((nens, ni*nj*nv))
  iStorm_ens = np.zeros(nens)
  jStorm_ens = np.zeros(nens)
  for n in range(nens):
    iStorm_ens[n] = iStorm + iBias + np.random.normal(0, 1) * Csprd
    jStorm_ens[n] = jStorm + jBias + np.random.normal(0, 1) * Csprd
    Rmw_n = Rmw + np.random.normal(0, 1) * Rsprd
    Vmax_n = Vmax + np.random.normal(0, 1) * Vsprd
    Vout_n = Vout + np.random.normal(0, 1) * 0.0
    Xb[n, :] = rv.make_state(ni, nj, nv, iStorm_ens[n], jStorm_ens[n], Rmw_n, Vmax_n, Vout_n)

  ###Observations
  th = np.random.uniform(0, 360)*np.pi/180
  io = iStorm + Rmw*np.sin(th)
  jo = iStorm + Rmw*np.cos(th)
  # print(io, jo)
  iObs = np.array([io, io])
  jObs = np.array([jo, jo])
  vObs = np.array([0, 1])
  nobs = iObs.size   # number of observation points
  obserr = 1 # observation error spread
  L = rv.location_operator(iX, jX, iObs, jObs)
  H = rv.obs_operator(iX, jX, nv, iObs, jObs, vObs)
  obs = np.matmul(H, Xt) + np.random.normal(0.0, obserr, nobs)

  ##Run filter
  Xa[realize, :, :] = Xb.copy()
  Yb = np.matmul(H, Xb.T)
  if filter_kind == 'EnSRF':
    Xa[realize, :, :] = DA.EnSRF(ni, nj, nv, Xb, Yb, iX, jX, H, iObs, jObs, vObs, obs, obserr, localize_cutoff)
  if filter_kind == 'PF':
    Xa[realize, :, :] = DA.PF(ni, nj, nv, Xb, Yb, iX, jX, H, iObs, jObs, vObs, obs, obserr, localize_cutoff)
  if filter_kind == 'EnSRF_MSA':
    krange = (1,2,3,4)
    ns = len(krange)
    X = Xb.copy()
    for s in range(ns):
      Xsb = X.copy()
      for m in range(nens):
        Xsb[m, :] = DA.get_scale(ni, nj, nv, X[m, :], krange, s)
      Y = np.matmul(H, X.T)
      Xsa = DA.EnSRF(ni, nj, nv, Xsb, Y, iX, jX, H, iObs, jObs, vObs, obs, obserr, localize_cutoff)
      if s < ns-1:
        for v in range(nv):
          xb = np.zeros((ni, nj, nens))
          xa = np.zeros((ni, nj, nens))
          xv = np.zeros((ni, nj, nens))
          for m in range(nens):
            xb[:, :, m] = np.reshape(Xsb[m, v*ni*nj:(v+1)*ni*nj], (ni, nj))
            xa[:, :, m] = np.reshape(Xsa[m, v*ni*nj:(v+1)*ni*nj], (ni, nj))
            xv[:, :, m] = np.reshape(X[m, v*ni*nj:(v+1)*ni*nj], (ni, nj))
          qu, qv = DA.optical_flow_HS(xb, xa, 5)
          xv = DA.warp(xv, -qu, -qv)
          # xv += xa - DA.warp(xb, -qu, -qv)
          for m in range(nens):
            X[m, v*ni*nj:(v+1)*ni*nj] = np.reshape(xv[:, :, m], (ni*nj,))
      else:
        X += Xsa - Xsb
    Xa[realize, :, :] = X

np.save('/run/media/yingyue/BACKUP2020/tmp/{}_Csprd{}_N{}'.format(filter_kind, Csprd, nens), Xa)

##diagnose & plot
# plt.switch_backend('Agg')
# plt.figure(figsize=(5, 5))
# ax = plt.subplot(1, 1, 1)
# # ax.scatter(iStorm_ens,jStorm_ens, s=3, color='.7')
# cmap = [plt.cm.jet(x) for x in np.linspace(0, 1,nens)]
# for n in range(nens):
#   # ax.scatter(iStorm_ens[n],jStorm_ens[n], s=40, color=[cmap[n][0:3]])
#   g.plot_contour(ax,ni,nj, Xa[n, :], [cmap[n][0:3]], 1)
# g.plot_contour(ax, ni, nj, Xt, 'black', 3)
# ax.plot(iObs, jObs, 'kx')
# g.set_axis(ax,ni,nj)
# ax.tick_params(labelsize=15)
# # plt.savefig('1.png', dpi=100)
# plt.show()

