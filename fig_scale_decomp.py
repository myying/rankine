import numpy as np
import matplotlib.pyplot as plt
from rankine_vortex import *
from data_assimilation import *
from multiscale import *
from config import *

Vbg = 4
np.random.seed(100)
Xout = gen_random_flow(ni, nj, nv, dx, Vbg, -3) + gen_vortex(ni, nj, nv, Vmax, Rmw, 0)
ii, jj = np.mgrid[0:ni, 0:nj]
iout = (ii-ni/2)*dx/1000
jout = (jj-nj/2)*dx/1000

ns = 3
Xsc = np.zeros((ni, nj, nv, ns))

plt.switch_backend('Agg')
plt.figure(figsize=(14,9))

ax = plt.subplot(222)
c = ax.contourf(iout, jout, Xout[:, :, 0], np.arange(-40, 41, 4), cmap='bwr')
ax.set_aspect('equal')
ax.tick_params(labelsize=13)
cb = plt.colorbar(c)
cb.ax.tick_params(labelsize=12)

ax1 = plt.subplot(221)
wn, pwr = pwr_spec(Xout)
ax1.loglog(wn, pwr, color='black', linewidth=3, label='total')

for s in range(ns):
    Xs = get_scale(Xout, get_krange(ns), s)
    wn, pwr = pwr_spec(Xs)
    ax = plt.subplot(2,3,s+4)
    ax.tick_params(labelsize=13)
    c = ax.contourf(iout, jout, Xs[:, :, 0], np.arange(-20, 21, 2), cmap='bwr')
    cb = plt.colorbar(c)
    cb.ax.tick_params(labelsize=12)
    ax1.loglog(wn, pwr, linewidth=2, label='s={}'.format(s+1))
ax1.set_xlim([1, 40])
ax1.set_ylim([1e-3, 1e2])
ax1.legend()
ax1.tick_params(labelsize=12)


plt.savefig('out.pdf')
