#!/usr/bin/env python
import numpy as np
import rankine_vortex as rv
import sys

ni = 128  # number of grid points i, j directions
nj = 128
nv = 2   # number of variables, (u, v)
filter_kind = sys.argv[1]
nens = int(sys.argv[2])
Csprd = int(sys.argv[3])
obsR = int(sys.argv[4])

### Rankine Vortex definition, truth
Rmw = 5    # radius of maximum wind
Vmax = 50   # maximum wind speed
Vout = 0    # wind speed outside of vortex
iStorm = 63 # location of vortex in i, j
jStorm = 63

##truth
iX, jX = rv.make_coords(ni, nj)
Xt = rv.make_state(ni, nj, nv, iStorm, jStorm, Rmw, Vmax, Vout)

Xa = np.load('out/{}_Csprd{}_N{}_obsR{}.npy'.format(filter_kind, Csprd, nens, obsR))
nrealize, nm, nx = Xa.shape

###domain-averaged state error:
rmse = np.zeros(nrealize)
for r in range(nrealize):
  rmse[r] = np.sqrt(np.mean((np.mean(Xa[r, :, :], axis=0)-Xt)**2))
np.save('out/diag/mean_error/{}_Csprd{}_N{}_obsR{}.npy'.format(filter_kind, Csprd, nens, obsR), rmse)

# rmse1 = np.sqrt(np.mean((np.mean(Xa, axis=0) - Xt)**2))
# rmse2 = 0.0
# for m in range(nens):
#   rmse2 += np.mean((Xa[m, :] - Xt)**2)
# rmse2 = np.sqrt(rmse2/float(nens))

###intensity track
# utrue, vtrue = rv.X2uv(ni, nj, Xt)
# wtrue = rv.get_max_wind(utrue, vtrue)
# itrue, jtrue = rv.get_center_ij(utrue, vtrue)
# umean, vmean = rv.X2uv(ni, nj, np.mean(Xa, axis=0))
# wmean = rv.get_max_wind(umean, vmean)
# imean, jmean = rv.get_center_ij(umean, vmean)
# wmem = np.zeros(nens)
# imem = np.zeros(nens)
# jmem = np.zeros(nens)
# for m in range(nens):
#   umem, vmem = rv.X2uv(ni, nj, Xa[m, :])
#   wmem[m] = rv.get_max_wind(umem, vmem)
#   imem[m], jmem[m] = rv.get_center_ij(umem, vmem)
# print(wtrue, wmean, np.mean(wmem))
# print(itrue, imean, np.mean(imem))
# print(jtrue, jmean, np.mean(jmem))
