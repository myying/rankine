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
Csprd = 3
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
# th = np.random.uniform(0, 360)*np.pi/180 * np.array([1, 1])
th = 45*np.pi/180 * np.array([1])
iObs = iStorm + Rmw*np.sin(th)
jObs = iStorm + Rmw*np.cos(th)
vObs = np.array([0])
nobs = iObs.size   # number of observation points
obserr = 3 # observation error spread
H = rv.obs_operator(iX, jX, nv, iObs, jObs, vObs)
obs = np.dot(H, Xt) + np.random.normal(0.0, obserr, nobs)

##Run filter
Xa = np.zeros((4, nens, ni*nj*nv))
Yb = np.dot(H, Xb.T)
Xa[0, :, :] = Xb
Xa[1, :, :] = DA.EnSRF(ni, nj, nv, Xb, Yb, iX, jX, H, iObs, jObs, vObs, obs, obserr, 0)
Xa[3, :, :] = DA.PF(ni, nj, nv, Xb, Yb, iX, jX, H, iObs, jObs, vObs, obs, obserr, 0)

krange = (1,2,3,4)
ns = len(krange)
X = Xb.copy()
for s in range(2):
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
Xa[2, :, :] = X

##plot
plt.switch_backend('Agg')
plt.figure(figsize=(12, 6))
ii, jj = np.mgrid[0:ni, 0:nj]
cmap = [plt.cm.jet(m) for m in np.linspace(0.2, 0.8, nens)]

iout = np.array([70])
jout = np.array([55])
vout = np.array([0])
Hout = rv.obs_operator(iX, jX, nv, iout, jout, vout)
title=('(a) Prior', '(b) EnSRF', '(c) EnSRF+MSA', '(d) PF')
for i in range(4):
  ax = plt.subplot(2, 4, i+1)
  for m in range(nens):
    u, v = rv.X2uv(ni, nj, Xa[i, m, :])
    ax.contour(ii, jj, u, (-15, 15), colors=[cmap[m][0:3]], linewidths=2)
  ut, vt = rv.X2uv(ni, nj, Xt)
  ax.contour(ii, jj, ut, (-15, 15), colors='k', linewidths=3)
  ax.plot(iObs, jObs, 'k+', markersize=10, markeredgewidth=2)
  ax.plot(iout, jout, 'ko', markersize=10, markeredgewidth=2)
  ax.set_aspect('equal', 'box')
  ax.set_xlim(43, 83)
  ax.set_ylim(43, 83)
  ax.set_xticks(np.arange(43, 84, 10))
  ax.set_xticklabels(np.arange(-20, 21, 10))
  ax.set_yticks(np.arange(43, 84, 10))
  ax.set_yticklabels(np.arange(-20, 21, 10))
  ax.tick_params(labelsize=12)
  ax.set_title(title[i], fontsize=16, loc='left')

for i in range(1, 4):
  ax = plt.subplot(2, 4, 5+i)
  obs = np.dot(H[0, :], Xa[0, :, :].T)
  state = np.dot(Hout[0, :], Xa[0, :, :].T)
  ax.scatter(obs, state, color='c')
  obs1 = np.dot(H[0, :], Xa[i, :, :].T)
  state1 = np.dot(Hout[0, :], Xa[i, :, :].T)
  for m in range(nens):
    ax.plot([obs[m], obs1[m]], [state[m], state1[m]], 'k', linewidth=0.5)
  ax.scatter(obs1, state1, color='y')
  ax.set_xlim(-60, 60)
  ax.set_ylim(-20, 60)
  ax.set_xlabel('obs (+)')
  ax.set_ylabel('state (o)')

plt.savefig('out/1.pdf')
