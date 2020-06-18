#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import rankine_vortex as rv

filter_kind = ('none', 'EnSRF', 'EnSRF_MSA', 'PF')
colors = ((0.5, 0.5, 0.5), (0.5, 0.5, 1.0), (1.0, 0.5, 0.5), (0.5, 1.0, 0.5))
localize_cutoff = 0

ni = 128  # number of grid points i, j directions
nj = 128
nv = 2   # number of variables, (u, v)
nens = (20,) # ensemble size
Csprd = (3,)

### Rankine Vortex definition, truth
Rmw = 5    # radius of maximum wind
Vmax = 50   # maximum wind speed
Vout = 0    # wind speed outside of vortex
iStorm = 63 # location of vortex in i, j
jStorm = 63

##truth
iX, jX = rv.make_coords(ni, nj)
Xt = rv.make_state(ni, nj, nv, iStorm, jStorm, Rmw, Vmax, Vout)

nrealize = 100

###domain-averaged state error:
rmse = np.zeros((nrealize, len(filter_kind), len(nens), len(Csprd)))
for k in range(len(filter_kind)):
  for l in range(len(Csprd)):
    for n in range(len(nens)):
      Xa = np.load('out/{}_Csprd{}_N{}.npy'.format(filter_kind[k], Csprd[l], nens[n]))
      for r in range(nrealize):
        rmse[r, k, n, l] = np.sqrt(np.mean((np.mean(Xa[r, :, :], axis=0)-Xt)**2))

plt.switch_backend('Agg')
plt.figure(figsize=(5, 5))
ax = plt.subplot(1, 1, 1)
for l in range(len(Csprd)):
  for n in range(len(nens)):
    for k in range(len(filter_kind)):
      bx = ax.boxplot(rmse[:, k, n, l], positions=[(k+1)], widths=0.2, patch_artist=True)
      for item in ['boxes', 'whiskers', 'medians', 'caps']:
        plt.setp(bx[item], color='k')
      plt.setp(bx['boxes'], facecolor=colors[k])
ax.set_xlim(0, 5)
ax.set_xticks((1, 2, 3, 4))
ax.set_xticklabels(('Prior', 'EnSRF', 'EnSRF+MSA', 'PF'))
plt.savefig('1.pdf')

# rmse1 = np.sqrt(np.mean((np.mean(Xa, axis=0) - Xt)**2))
# rmse2 = 0.0
# for m in range(nens):
#   rmse2 += np.mean((Xa[m, :] - Xt)**2)
# rmse2 = np.sqrt(rmse2/float(nens))
# print(rmse2, rmse2)

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
