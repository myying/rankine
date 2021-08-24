#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from rankine_vortex import *
from obs_def import *
from data_assimilation import *
from multiscale import *
from config import *
import sys

##truth
Xt = gen_vortex(ni, nj, nv, Vmax, Rmw, 0)

expdir =sys.argv[1]
casename = sys.argv[2]
Yo = np.load(outdir+expdir+'/Yo.npy')
Yloc = np.load(outdir+expdir+'/Yloc.npy')
X = np.load(outdir+expdir+'/'+casename+'.npy')

plt.figure(figsize=(12, 10))
ii, jj = np.mgrid[0:ni, 0:nj]
cmap = [plt.cm.jet(m) for m in np.linspace(0.2, 0.8, nens)]

ax = plt.subplot(111)
for m in range(nens):
    wspd = np.sqrt(X[:, :, 0, m]**2+X[:, :, 1, m]**2)
    ax.contour(ii, jj, wspd, (20,), colors=[cmap[m][0:3]], linewidths=2)
wspd = np.sqrt(Xt[:, :, 0]**2+Xt[:, :, 1]**2)
ax.contour(ii, jj, wspd, (20,), colors='k', linewidths=3)
ax.plot(Yloc[0, ::2], Yloc[1, ::2], 'k+', markersize=10, markeredgewidth=2)
ax.set_aspect('equal', 'box')
ax.set_xlim(0.5*ni-15, 0.5*ni+15)
ax.set_ylim(0.5*nj-15, 0.5*nj+15)
ax.set_xticks(np.arange(0.5*ni-15, 0.5*ni+16, 5))
ax.set_xticklabels(np.arange(-15, 16, 5))
ax.set_yticks(np.arange(0.5*nj-15, 0.5*nj+16, 5))
ax.set_yticklabels(np.arange(-15, 16, 5))
ax.tick_params(labelsize=12)

plt.show()
