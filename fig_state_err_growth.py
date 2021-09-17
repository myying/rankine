###run ensemble forecast with perturbed initial conditions: location spread in vortices
###and background flow noises 30% of flow amplitude.

import numpy as np
import matplotlib.pyplot as plt
from rankine_vortex import *
from data_assimilation import *
from config import *

nt = 13
loc_sprd = 1
Vbg_ens = 0.3*Vbg
gen_ens = np.random.uniform(2, 6, nens)*1e-5

np.random.seed(0)
bkg_flow = gen_random_flow(ni, nj, nv, dx, Vbg, -3)

true_center = np.zeros((2, nt))
true_intensity = np.zeros(nt)
ens_center = np.zeros((2, nens, nt))
ens_intensity = np.zeros((nens, nt))
mean_err = np.zeros(nt+1)
mean_sprd = np.zeros(nt+1)

##run truth simulation
Xt = np.zeros((ni, nj, nv, nt+1))
Xt[:, :, :, 0] = bkg_flow + gen_vortex(ni, nj, nv, Vmax, Rmw, 0)
Xt[:, :, :, 0] = np.roll(np.roll(Xt[:, :, :, 0], -20, axis=0), -20, axis=1)
for t in range(nt):
    Xt[:, :, :, t+1] = advance_time(Xt[:, :, :, t], dx, dt, smalldt, gen, diss)
    true_center[:, t] = vortex_center(Xt[:, :, :, t])
    true_intensity[t] = vortex_intensity(Xt[:, :, :, t])

##ensemble run
X = np.zeros((ni, nj, nv, nens, nt+1))
for m in range(nens):
    X[:, :, :, m, 0] = bkg_flow + gen_random_flow(ni, nj, nv, dx, Vbg_ens, -3) + gen_vortex(ni, nj, nv, Vmax, Rmw, loc_sprd)
X[:, :, :, :, 0] = np.roll(np.roll(X[:, :, :, :, 0], -20, axis=0), -20, axis=1)
for t in range(nt):
    print(t)
    X[:, :, :, :, t+1] = advance_time(X[:, :, :, :, t], dx, dt, smalldt, gen_ens, diss)
    for m in range(nens):
        ens_center[:, m, t] = vortex_center(X[:, :, :, m, t])
        ens_intensity[m, t] = vortex_intensity(X[:, :, :, m, t])

##plot summary
plt.switch_backend('Agg')
plt.figure(figsize=(12, 4))
cmap = [plt.cm.jet(x) for x in np.linspace(0, 1,nens)]

ax = plt.subplot(131)
for m in range(nens):
    ax.plot(ens_center[0, m, :], ens_center[1, m, :], color=cmap[m][0:3])
ax.plot(true_center[0, :], true_center[1, :], color='black', linewidth=3, label='truth')
ax.set_xlim([0, ni])
ax.set_ylim([0, nj])
ax.set_xticks(np.arange(0, 1001, 200)/dx*1000)
ax.set_yticks(np.arange(0, 1001, 200)/dx*1000)
ax.set_xticklabels(np.arange(0, 1001, 200))
ax.set_yticklabels(np.arange(0, 1001, 200))
ax.legend()
ax.tick_params(labelsize=12)

ax = plt.subplot(132)
tt = np.arange(0, nt)
for m in range(nens):
    ax.plot(tt, ens_intensity[m, :], color=cmap[m][0:3])
ax.plot(tt, true_intensity[:], color='black', linewidth=3, label='truth')
ax.set_xlim([0, nt])
ax.set_xticks(np.arange(0, nt+1, 2))
ax.set_ylim([0, 100])
ax.legend()
ax.tick_params(labelsize=12)

ax = plt.subplot(133)
tt = np.arange(0, nt+1)
ax.plot(tt, mean_rmse(X, Xt), color='black', linewidth=3, label='error')
ax.plot(tt, ens_sprd(X), color='black', linestyle=':', linewidth=3, label='sprd')
ax.set_xlim([0, nt])
ax.set_xticks(np.arange(0, nt+1, 2))
ax.set_ylim([0, 12])
ax.legend()
ax.tick_params(labelsize=12)

plt.savefig('out.pdf')
