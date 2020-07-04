#!/usr/bin/env python
import numpy as np
import rankine_vortex as rv
import matplotlib.pyplot as plt
import data_assimilation as DA

ni = 128  # number of grid points i, j directions
nj = 128
nv = 2   # number of variables, (u, v)
nens = 80 # ensemble size

### Rankine Vortex definition, truth
Rmw = 5    # radius of maximum wind
Vmax = 50   # maximum wind speed
Vout = 0    # wind speed outside of vortex
iStorm = 63 # location of vortex in i, j
jStorm = 63

##truth
iX, jX = rv.make_coords(ni, nj)
Xt = rv.make_state(ni, nj, nv, iStorm, jStorm, Rmw, Vmax, Vout)

np.random.seed(9)

##Prior ensemble
Csprd = 3.0
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
iObs = np.array([66.53, 66.53])
jObs = np.array([66.53, 66.53])
vObs = np.array([0, 1])
nobs = iObs.size   # number of observation points
obserr = 4.0 # observation error spread
H = rv.obs_operator(iX, jX, nv, iObs, jObs, vObs)
obs = np.dot(H, Xt) + np.array([2.0, -2.0])

iout = np.array([70])
jout = np.array([55])
vout = np.array([0])
Hout = rv.obs_operator(iX, jX, nv, iout, jout, vout)

##Run filter
Xa1 = np.zeros((2, nens, ni*nj*nv))
Xa1[0, :, :] = DA.EnSRF(ni, nj, nv, Xb, np.dot(H[0:1, :], Xb.T), iX, jX, H[0:1, :], iObs[0:1], jObs[0:1], vObs[0:1], obs[0:1], obserr, 0)
Xa1[1, :, :] = DA.EnSRF(ni, nj, nv, Xa1[0, :, :], np.dot(H[1:2, :], Xa1[0, :, :].T), iX, jX, H[1:2, :], iObs[1:2], jObs[1:2], vObs[1:2], obs[1:2], obserr, 0)

ns = 8
krange = np.arange(1, ns+1)
Xa2 = np.zeros((ns, nens, ni*nj*nv))
ns = len(krange)
X = Xb.copy()
for s in range(ns):
  Xsb = X.copy()
  for m in range(nens):
    Xsb[m, :] = DA.get_scale(ni, nj, nv, X[m, :], krange, s)
  Y = np.dot(H, X.T)
  Xsa = DA.EnSRF(ni, nj, nv, Xsb, Y, iX, jX, H, iObs, jObs, vObs, obs, obserr, 0)
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
      xv += xa - DA.warp(xb, -qu, -qv)
      for m in range(nens):
        X[m, v*ni*nj:(v+1)*ni*nj] = np.reshape(xv[:, :, m], (ni*nj,))
  else:
    X += Xsa - Xsb
  Xa2[s, :, :] = X.copy()

Xa3 = DA.PF(ni, nj, nv, Xb, np.dot(H, Xb.T), iX, jX, H, iObs, jObs, vObs, obs, obserr, 0)

##plot
plt.switch_backend('Agg')
plt.figure(figsize=(10, 10))

for i in range(2):
  ax = plt.subplot(3, 2, i+1)
  obs_truth = np.dot(H[i, :], Xt.T)
  ax.plot([obs_truth, obs_truth], [0, 40], color=[.7, .7, .7], linewidth=2, zorder=-1)
  state_truth = np.dot(Hout[0, :], Xt.T)
  ax.plot([-50, 50], [state_truth, state_truth], color=[.7, .7, .7], linewidth=2, zorder=-1)
  ax.plot([obs[i], obs[i]], [0, 40], 'r', linewidth=2, zorder=-1)
  obs_out = np.dot(H[i, :], Xb.T)
  state_out = np.dot(Hout[0, :], Xb.T)
  ax.scatter(obs_out, state_out, s=50, color=[.3, .7, .3], alpha=0.7, edgecolor='None', zorder=0)
  obs_out1 = np.dot(H[i, :], Xa1[0, :, :].T)
  state_out1 = np.dot(Hout[0, :], Xa1[0, :, :].T)
  obs_out2 = np.dot(H[i, :], Xa1[1, :, :].T)
  state_out2 = np.dot(Hout[0, :], Xa1[1, :, :].T)
  ax.scatter(obs_out2, state_out2, s=50, color=[0, 0, 1], alpha=0.7, edgecolor='None', zorder=0)
  for m in range(30):
    ax.plot([obs_out[m], obs_out1[m]], [state_out[m], state_out1[m]], 'k', linewidth=0.7, zorder=1)
    ax.plot([obs_out1[m], obs_out2[m]], [state_out1[m], state_out2[m]], 'k', linewidth=0.7, zorder=1)
  ax.set_ylim(0, 40)
  if i == 0:
    ax.set_xlim(-50, 40)
  if i == 1:
    ax.set_xlim(-20, 50)

for i in range(2):
  ax = plt.subplot(3, 2, i+3)
  obs_truth = np.dot(H[i, :], Xt.T)
  ax.plot([obs_truth, obs_truth], [0, 40], color=[.7, .7, .7], linewidth=2, zorder=-1)
  state_truth = np.dot(Hout[0, :], Xt.T)
  ax.plot([-50, 50], [state_truth, state_truth], color=[.7, .7, .7], linewidth=2, zorder=-1)
  ax.plot([obs[i], obs[i]], [0, 40], 'r', linewidth=2, zorder=-1)
  obs_out = np.dot(H[i, :], Xb.T)
  state_out = np.dot(Hout[0, :], Xb.T)
  ax.scatter(obs_out, state_out, s=50, color=[.3, .7, .3], alpha=0.7, edgecolor='None', zorder=0)
  obs_out1 = np.zeros((ns, nens))
  state_out1 = np.zeros((ns, nens))
  for s in range(ns):
    obs_out1[s, :] = np.dot(H[i, :], Xa2[s, :, :].T)
    state_out1[s, :] = np.dot(Hout[0, :], Xa2[s, :, :].T)
  ax.scatter(obs_out1[-1, :], state_out1[-1, :], s=50, color=[0, 0, 1], alpha=0.7, edgecolor='None', zorder=0)
  for m in range(30):
    ax.plot([obs_out[m], obs_out1[0, m]], [state_out[m], state_out1[0, m]], 'k', linewidth=0.7, zorder=1)
    for s in range(1, ns):
      ax.plot([obs_out1[s-1, m], obs_out1[s, m]], [state_out1[s-1, m], state_out1[s, m]], 'k', linewidth=0.7, zorder=1)
  ax.set_ylim(0, 40)
  if i == 0:
    ax.set_xlim(-50, 40)
  if i == 1:
    ax.set_xlim(-20, 50)

for i in range(2):
  ax = plt.subplot(3, 2, i+5)
  obs_truth = np.dot(H[i, :], Xt.T)
  ax.plot([obs_truth, obs_truth], [0, 40], color=[.7, .7, .7], linewidth=2, zorder=-1)
  state_truth = np.dot(Hout[0, :], Xt.T)
  ax.plot([-50, 50], [state_truth, state_truth], color=[.7, .7, .7], linewidth=2, zorder=-1)
  ax.plot([obs[i], obs[i]], [0, 40], 'r', linewidth=2, zorder=-1)
  obs_out = np.dot(H[i, :], Xb.T)
  state_out = np.dot(Hout[0, :], Xb.T)
  ax.scatter(obs_out, state_out, s=50, color=[.3, .7, .3], alpha=0.7, edgecolor='None', zorder=0)
  obs_out1 = np.dot(H[i, :], Xa3.T)
  state_out1 = np.dot(Hout[0, :], Xa3.T)
  ax.scatter(obs_out1, state_out1, s=50, color=[0, 0, 1], alpha=0.7, edgecolor='None', zorder=0)
  for m in range(30):
    ax.plot([obs_out[m], obs_out1[m]], [state_out[m], state_out1[m]], 'k', linewidth=0.7, zorder=1)
  ax.set_ylim(0, 40)
  if i == 0:
    ax.set_xlim(-50, 40)
  if i == 1:
    ax.set_xlim(-20, 50)

plt.savefig('Fig02.pdf')
