#!/usr/bin/env python
import numpy as np
import rankine_vortex as rv
import alignment as al
import DA
import config as p

###Data assimilation trial
# Xa = DA.EnSRF(ni, nj, nv, Xb, iX, jX, H, iObs, jObs, obs, obserr, localize_cutoff)

# Xa = DA.LPF(ni, nj, nv, Xb, iX, jX, H, iObs, jObs, obs, obserr, localize_cutoff, alpha)

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
