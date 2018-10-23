#!/usr/bin/env python2

import numpy as np
import rankine_vortex as rv
import alignment as al
import graphics as g
import matplotlib.pyplot as plt
import DA

plt.switch_backend('Agg')
plt.figure(figsize=(8, 8))

np.random.seed(0)  # fix random number seed, make results predictable

ni = 41  # number of grid points i, j directions
nj = 41
nv = 2   # number of variables, (u, v)
nens = 20  # ensemble size

### Rankine Vortex definition, truth
Rmw = 4    # radius of maximum wind
Vmax = 35   # maximum wind speed
Vout = 3    # wind speed outside of vortex
iStorm = 20 # location of vortex in i, j
jStorm = 20

nobs = 100   # number of observations
obserr = 1.0 # observation error spread
localize_cutoff = 100  # localization cutoff distance (taper to zero)
alpha = 0.5

iX, jX = rv.make_coords(ni, nj)

Xt = rv.make_state(ni, nj, nv, iStorm, jStorm, Rmw, Vmax, Vout)


Xb = np.zeros((nens, ni*nj*nv))
Csprd = 0.7*Rmw
iBias = 0
jBias = 0
Rsprd = 2
Vsprd = 3
for n in range(nens):
  iStorm_n = iStorm + iBias + np.random.normal(0, 1) * Csprd
  jStorm_n = jStorm + jBias + np.random.normal(0, 1) * Csprd
  Rmw_n = Rmw + np.random.normal(0, 1) * Rsprd
  Vmax_n = Vmax + np.random.normal(0, 1) * Vsprd
  Vout_n = Vout + np.random.normal(0, 1) * 0.0
  Xb[n, :] = rv.make_state(ni, nj, nv, iStorm_n, jStorm_n, Rmw_n, Vmax_n, Vout_n)
  # iD, jD = al.random_vector(ni, nj, np.array([0, 0]), 20, 3)
  # for v in range(nv):
  #   Xb[n, v*ni*nj:(v+1)*ni*nj] = al.deformation(ni, nj, Xb[n, v*ni*nj:(v+1)*ni*nj], iD, jD)
ax = plt.subplot(2, 2, 1)
g.plot_ens(ax, ni, nj, Xb, Xt)
g.set_axis(ax, ni, nj)
ax.set_title('Prior members')


iObs = np.random.uniform(0, ni, size=nobs)
jObs = np.random.uniform(0, nj, size=nobs)
L = rv.location_operator(iX, jX, iObs, jObs)
iSite = 2
jSite = 2
H = rv.obs_operator(iX, jX, nv, iObs, jObs, iSite, jSite)
obs = np.matmul(H, Xt) + np.random.normal(0.0, obserr, nobs)
ax = plt.subplot(2, 2, 2)
g.plot_obs(ax, iObs, jObs, obs)
g.set_axis(ax, ni, nj)
ax.set_title('Obs: Vr')


###show before/after displacement
# iD, jD = al.random_vector(ni, nj, np.array([0, 0]), 100, 2) 
# ax = plt.subplot(2, 2, 3)
# ax.contourf(np.reshape(iX, (ni, nj)), np.reshape(jX, (ni, nj)), np.reshape(Xb[0, 0:ni*nj], (ni, nj)))
# iD = -4.5
# jD = -2.5
# iD, jD = al.displace_vector(ni, nj, nv, Xb[0, :], H, obs, obserr)
# ax = plt.subplot(2, 2, 4)
# c = ax.contourf(np.reshape(iX, (ni, nj)), np.reshape(jX, (ni, nj)), np.reshape(al.deformation(ni, nj, Xb[0, 0:ni*nj], iD, jD), (ni, nj)))


###show displaced member and its cost function for disp vector
# n = 0
# ax = plt.subplot(2, 2, 3)
# g.plot_ens(ax, ni, nj, Xb[n:n+1, :], Xt)
# g.set_axis(ax, ni, nj)
# # map of J over displacement in i and j
# aa, bb = np.mgrid[-20:20:1, -20:20:1]
# na, nb = aa.shape
# J = np.zeros((na, nb))
# for i in range(na):
#   for j in range(nb):
#     J[i, j] = al.cost_function(ni, nj, nv, Xb[n, :], H, obs, obserr, aa[i, j], bb[i, j])
# ax = plt.subplot(2,2,4)
# ax.contourf(aa, bb, J)


Xa = DA.EnSRF(ni, nj, nv, Xb, iX, jX, H, iObs, jObs, obs, obserr, localize_cutoff)
ax = plt.subplot(2, 2, 3)
g.plot_ens(ax, ni, nj, Xa, Xt)
g.set_axis(ax, ni, nj)
ax.set_title('EnSRF members')
# g.output_ens('1.nc', ni, nj, Xa, Xt)


# Xa = DA.LPF(ni, nj, nv, Xb, iX, jX, H, iObs, jObs, obs, obserr, localize_cutoff, alpha)
# ax = plt.subplot(2, 2, 3)
# g.plot_ens(ax, ni, nj, Xa, Xt)
# g.set_axis(ax, ni, nj)
# ax.set_title('LPF members')

###Find displacement vecot by minimization of cost function J
Xb1 = np.zeros((nens, ni*nj*nv))
for n in range(nens):
  nm = 9
  nit = 2
  iD = 0
  jD = 0
  for t in range(nit):
    nis = ni/nm**t
    njs = nj/nm**t
    iDens = np.zeros(nm*nm)
    jDens = np.zeros(nm*nm)
    J = np.zeros(nm*nm)
    m = 0
    for i in np.linspace(-nis/2, nis/2, nm):
      for j in np.linspace(-njs/2, njs/2, nm):
        iDens[m] = iD + i
        jDens[m] = jD + j
        J[m] = al.cost_function(ni, nj, nv, Xb[n, :], H, obs, obserr, iDens[m], jDens[m])
        m += 1
    w = np.exp(-J)
    if np.sum(w) == 0:
      w[np.where(J==min(J))] = 1
    w = w / np.sum(w)
    iD = np.sum(iDens*w)
    jD = np.sum(jDens*w)
    # iD = iDens[np.where(J==min(J))]
    # jD = jDens[np.where(J==min(J))]
  print((n, iD, jD))
  for v in range(nv):
    Xb1[n, v*ni*nj:(v+1)*ni*nj] = al.deformation(ni, nj, Xb[n, v*ni*nj:(v+1)*ni*nj], iD, jD)

####run EnSRF on aligned members
# Xa1 = DA.EnSRF(ni, nj, nv, Xb1, iX, jX, H, iObs, jObs, obs, obserr, localize_cutoff)
ax = plt.subplot(2, 2, 4)
g.plot_ens(ax, ni, nj, Xb1, Xt)
g.set_axis(ax, ni, nj)
ax.set_title('Aligned members')
# ax = plt.subplot(2, 2, 4)
# g.plot_ens(ax, ni, nj, Xa1, Xt)
# g.set_axis(ax, ni, nj)
# ax.set_title('Aligned+EnSRF members')
# g.output_ens('2.nc', ni, nj, Xa1, Xt)


plt.savefig('1.pdf')
