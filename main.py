#!/usr/bin/env python

import numpy as np
import rankine_vortex as rv
import alignment as al
import graphics as g
import matplotlib.pyplot as plt
import DA
import sys

plt.switch_backend('Agg')
plt.figure(figsize=(5, 5))

np.random.seed(1)  # fix random number seed, make results predictable

ni = 41  # number of grid points i, j directions
nj = 41
nv = 2   # number of variables, (u, v)
nens = 1000  # ensemble size
nens_show = 20

### Rankine Vortex definition, truth
Rmw = 5    # radius of maximum wind
Vmax = 30   # maximum wind speed
Vout = 0    # wind speed outside of vortex
iStorm = 20 # location of vortex in i, j
jStorm = 20
x_in = int(sys.argv[1])
y_in = int(sys.argv[2])
iout = np.array([x_in])
jout = np.array([y_in])
ioff = 0
joff = 12

nobs = 500   # number of observations
obserr = 1.0 # observation error spread
localize_cutoff = 100  # localization cutoff distance (taper to zero)
alpha = 0.5  ##for LPF

iX, jX = rv.make_coords(ni, nj)

Xt = rv.make_state(ni, nj, nv, iStorm, jStorm, Rmw, Vmax, Vout)

###prior ensemble
Xb = np.zeros((nens, ni*nj*nv))
Csprd = 0.1*Rmw
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
# ax = plt.subplot(1, 2, 1)
# g.plot_ens(ax, ni, nj, Xb, Xt)
# g.set_axis(ax, ni, nj)
# # ax.plot(iout, jout, 'wo')
# # ax.tick_params(labelsize=15)

###observations (radial velocity)
iObs = np.random.uniform(0, ni, size=nobs)
jObs = np.random.uniform(0, nj, size=nobs)
L = rv.location_operator(iX, jX, iObs, jObs)
iSite = 2
jSite = 2
H = rv.obs_operator(iX, jX, nv, iObs, jObs, iSite, jSite)
obs = np.matmul(H, Xt) + np.random.normal(0.0, obserr, nobs)

###plot obs
# ax = plt.subplot(1, 2, 2)
# g.plot_obs(ax, iObs, jObs, obs)
# g.set_axis(ax, ni, nj)
# g.plot_wind_contour(ax, ni, nj, Xt, 'black', 4)
# ax.tick_params(labelsize=15)

###plot histogram
# ax = plt.subplot(1, 1, 1)
# Hout = rv.obs_operator(iX, jX, nv, iout, jout, iSite, jSite)
# prior_err = np.dot(Hout, Xb.T) - np.dot(Hout, Xt)
# err_mean = np.mean(prior_err)
# err_std = np.std(prior_err)
# ii = np.arange(-50, 50, 1)
# jj = np.exp(-0.5*(ii-err_mean)**2/ err_std**2) / np.sqrt(2*np.pi) / err_std
# jj0 = g.hist_normal(ii, prior_err[0, :])
# ax.plot(ii, jj0, 'k', linewidth=4, label='Sample')
# ax.plot(ii, jj, 'r:', linewidth=2, label='Gaussian')
# ax.legend(fontsize=12, loc=1)
# ax.set_xlim(-30, 50)
# ax.set_ylim(-0.05, 0.5)
# ax.tick_params(labelsize=15)

# ax = plt.subplot(2, 3, 5)
# Hout = rv.obs_operator(iX, jX, nv, iout+ioff, jout+joff, iSite, jSite)
# prior_err = np.dot(Hout, Xb.T) - np.dot(Hout, Xt)
# err_mean = np.mean(prior_err)
# err_std = np.std(prior_err)
# ii = np.arange(-50, 50, 1)
# jj = np.exp(-0.5*(ii-err_mean)**2/ err_std**2) / np.sqrt(2*np.pi) / err_std
# jj0 = g.hist_normal(ii, prior_err[0, :])
# ax.plot(jj0, ii, 'k', linewidth=4, label='Sample')
# ax.plot(jj, ii, 'r:', linewidth=2, label='Gaussian')
# ax.legend(fontsize=12, loc=1)
# ax.set_ylim(-30, 50)
# ax.tick_params(labelsize=15)

###Covariance structure
ax = plt.subplot(1, 1, 1)
Hout = rv.obs_operator(iX, jX, nv, iout, jout, iSite, jSite)
x1 = np.dot(Hout, Xb.T)
corr_map = np.zeros((ni, nj))
for n in range(ni*nj):
  i = iX[n]
  j = jX[n]
  x2 = Xb[:, n]
  corr_map[i, j] = g.sample_correlation(x1, x2)
ii, jj = np.mgrid[0:ni, 0:nj]
ax.contourf(ii, jj, corr_map, np.arange(-1, 1.2, 0.1), cmap='bwr')
print(corr_map[iout[0], jout[0]])
# print(corr_map[iout[0]+ioff, jout[0]+joff])
g.set_axis(ax, ni, nj)
g.plot_wind_contour(ax, ni, nj, Xt, 'black', 4)
ax.plot(iout, jout, 'k+', markersize=10)
# ax.plot(iout+ioff, jout+joff, 'co')
ax.tick_params(labelsize=15)

###scatter plot
# ax = plt.subplot(2, 3, 6)
# Hout = rv.obs_operator(iX, jX, nv, iout, jout, iSite, jSite)
# err1 = np.dot(Hout, Xb.T) - np.dot(Hout, Xt)
# Hout = rv.obs_operator(iX, jX, nv, iout+ioff, jout+joff, iSite, jSite)
# err2 = np.dot(Hout, Xb.T) - np.dot(Hout, Xt)
# ax.scatter(err1[0, :], err2[0, :], s=0.3, c='k')
# cmap = [plt.cm.jet(x) for x in np.linspace(0, 1, nens_show)]
# for n in range(nens_show):
#   ax.scatter(err1[0, n], err2[0, n], s=10, c=[cmap[n][0:3]])
# ax.set_xlim(-30, 50)
# ax.set_ylim(-30, 50)
# ax.tick_params(labelsize=15)

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


###Data assimilation trial
# print('Running EnSRF')
# Xa = DA.EnSRF(ni, nj, nv, Xb, iX, jX, H, iObs, jObs, obs, obserr, localize_cutoff)
# ax = plt.subplot(1, 3, 2)
# g.plot_ens(ax, ni, nj, Xa, Xt)
# g.set_axis(ax, ni, nj)
# ax.tick_params(labelsize=15)

# Xa = DA.LPF(ni, nj, nv, Xb, iX, jX, H, iObs, jObs, obs, obserr, localize_cutoff, alpha)
# ax = plt.subplot(2, 2, 3)
# g.plot_ens(ax, ni, nj, Xa, Xt)
# g.set_axis(ax, ni, nj)
# ax.set_title('LPF members')

###Find displacement vector by minimization of cost function J, use SPSA method
# print('Running Alignment')
# Xb1 = np.zeros((nens, ni*nj*nv))
# for n in range(nens):
#   iD = 0.0
#   jD = 0.0
#   niter = 100
#   a = 1.0/nobs
#   c = 1.0
#   alpha = 0.8
#   gamma = 0.8
#   Jop = 0.5*nobs
#   nc = 0
#   J0 = al.cost_function(ni, nj, nv, Xb[n, :], H, obs, obserr, iD, jD)
#   J00 = J0
#   for k in range(niter):
#     ak = a / (k+5)**alpha
#     ck = c / (k+1)**gamma
#     delta_i = 2*np.round(np.random.uniform(0, 1))-1
#     delta_j = 2*np.round(np.random.uniform(0, 1))-1
#     J = al.cost_function(ni, nj, nv, Xb[n, :], H, obs, obserr, iD, jD)
#     if abs(J-J0) < 0.01*Jop:
#       nc += 1
#     else:
#       nc = 0
#     if J < 1.3*Jop or nc > 10:
#       break
#     J0 = J
#     J1 = al.cost_function(ni, nj, nv, Xb[n, :], H, obs, obserr, iD+ck*delta_i, jD+ck*delta_j)
#     J2 = al.cost_function(ni, nj, nv, Xb[n, :], H, obs, obserr, iD-ck*delta_i, jD-ck*delta_j)
#     iG = (J1-J2) / (2*ck*delta_i)
#     jG = (J1-J2) / (2*ck*delta_j)
#     iD -= ak*iG
#     jD -= ak*jG
#     # print((J, J1-J, J2-J, iD, jD))
#   print('{:3d}, J={:7.2f} ->{:7.2f}, displace ({:-7.3f}, {:-7.3f})'.format(n, J00, J, iD, jD))
#   for v in range(nv):
#     Xb1[n, v*ni*nj:(v+1)*ni*nj] = al.deformation(ni, nj, Xb[n, v*ni*nj:(v+1)*ni*nj], iD, jD)

####run EnSRF on aligned members
# Xa1 = DA.EnSRF(ni, nj, nv, Xb1, iX, jX, H, iObs, jObs, obs, obserr, localize_cutoff)
# ax = plt.subplot(1, 3, 3)
# g.plot_ens(ax, ni, nj, Xb1, Xt)
# g.set_axis(ax, ni, nj)
# ax.set_title('Aligned members')
# g.plot_ens(ax, ni, nj, Xa1, Xt)
# g.set_axis(ax, ni, nj)
# ax.set_title('Aligned+EnSRF members')

##output netcdf files
# g.output_ens('1.nc', ni, nj, Xa, Xt)
# g.output_ens('2.nc', ni, nj, Xa1, Xt)

plt.savefig('error_correlation/{}_{}.png'.format(x_in, y_in), dpi=100)
# plt.savefig('1.pdf')

