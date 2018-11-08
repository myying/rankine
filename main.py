#!/usr/bin/env python2

import numpy as np
import rankine_vortex as rv
import alignment as al
import graphics as g
import matplotlib.pyplot as plt
import DA

plt.switch_backend('Agg')
plt.figure(figsize=(12, 4))

np.random.seed(12345)  # fix random number seed, make results predictable

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

nobs = 64   # number of observations
obserr = 1.0 # observation error spread
localize_cutoff = 100  # localization cutoff distance (taper to zero)
alpha = 0.5

iX, jX = rv.make_coords(ni, nj)

Xt = rv.make_state(ni, nj, nv, iStorm, jStorm, Rmw, Vmax, Vout)


Xb = np.zeros((nens, ni*nj*nv))
Csprd = 0.6*Rmw
iBias = 0
jBias = 0
Rsprd = 0
Vsprd = 0
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
ax = plt.subplot(1, 3, 1)
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
# ax = plt.subplot(2, 2, 2)
# g.plot_obs(ax, iObs, jObs, obs)
# g.set_axis(ax, ni, nj)
# ax.set_title('Obs: Vr')


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
# n = 14
# ax = plt.subplot(2, 2, 3)
# g.plot_ens(ax, ni, nj, Xb[n:n+1, :], Xt)
# g.set_axis(ax, ni, nj)
# ### map of J over displacement in i and j
# aa, bb = np.mgrid[-20:20:1, -20:20:1]
# na, nb = aa.shape
# J = np.zeros((na, nb))
# for i in range(na):
#   for j in range(nb):
#     J[i, j] = al.cost_function(ni, nj, nv, Xb[n, :], H, obs, obserr, aa[i, j], bb[i, j])
# ax = plt.subplot(2,2,4)
# ax.contourf(aa, bb, J)

print('Running EnSRF')
Xa = DA.EnSRF(ni, nj, nv, Xb, iX, jX, H, iObs, jObs, obs, obserr, localize_cutoff)
ax = plt.subplot(1, 3, 2)
g.plot_ens(ax, ni, nj, Xa, Xt)
g.set_axis(ax, ni, nj)
ax.set_title('EnKF members')
# g.output_ens('1.nc', ni, nj, Xa, Xt)


# Xa = DA.LPF(ni, nj, nv, Xb, iX, jX, H, iObs, jObs, obs, obserr, localize_cutoff, alpha)
# ax = plt.subplot(2, 2, 3)
# g.plot_ens(ax, ni, nj, Xa, Xt)
# g.set_axis(ax, ni, nj)
# ax.set_title('LPF members')

###Find displacement vector by minimization of cost function J, use SPSA method
print('Running Alignment')
Xb1 = np.zeros((nens, ni*nj*nv))
for n in range(nens):
  iD = 0.0
  jD = 0.0
  niter = 100
  a = 1.0/nobs
  c = 1.0
  alpha = 0.8
  gamma = 0.8
  Jop = 0.5*nobs
  nc = 0
  J0 = al.cost_function(ni, nj, nv, Xb[n, :], H, obs, obserr, iD, jD)
  J00 = J0
  for k in range(niter):
    ak = a / (k+5)**alpha
    ck = c / (k+1)**gamma
    delta_i = 2*np.round(np.random.uniform(0, 1))-1
    delta_j = 2*np.round(np.random.uniform(0, 1))-1
    J = al.cost_function(ni, nj, nv, Xb[n, :], H, obs, obserr, iD, jD)
    if abs(J-J0) < 0.01*Jop:
      nc += 1
    else:
      nc = 0
    if J < 1.3*Jop or nc > 10:
      break
    J0 = J
    J1 = al.cost_function(ni, nj, nv, Xb[n, :], H, obs, obserr, iD+ck*delta_i, jD+ck*delta_j)
    J2 = al.cost_function(ni, nj, nv, Xb[n, :], H, obs, obserr, iD-ck*delta_i, jD-ck*delta_j)
    iG = (J1-J2) / (2*ck*delta_i)
    jG = (J1-J2) / (2*ck*delta_j)
    iD -= ak*iG
    jD -= ak*jG
    # print((J, J1-J, J2-J, iD, jD))
  print('{:3d}, J={:7.2f} ->{:7.2f}, displace ({:-7.3f}, {:-7.3f})'.format(n, J00, J, iD, jD))
  for v in range(nv):
    Xb1[n, v*ni*nj:(v+1)*ni*nj] = al.deformation(ni, nj, Xb[n, v*ni*nj:(v+1)*ni*nj], iD, jD)

####run EnSRF on aligned members
# Xa1 = DA.EnSRF(ni, nj, nv, Xb1, iX, jX, H, iObs, jObs, obs, obserr, localize_cutoff)
ax = plt.subplot(1, 3, 3)
g.plot_ens(ax, ni, nj, Xb1, Xt)
g.set_axis(ax, ni, nj)
ax.set_title('Aligned members')
# g.plot_ens(ax, ni, nj, Xa1, Xt)
# g.set_axis(ax, ni, nj)
# ax.set_title('Aligned+EnSRF members')
# g.output_ens('2.nc', ni, nj, Xa1, Xt)


plt.savefig('1.pdf')
