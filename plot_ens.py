#!/usr/bin/env python

import numpy as np
import rankine_vortex as rv
import graphics as g
import config as p
import matplotlib.pyplot as plt

plt.switch_backend('Agg')
plt.figure(figsize=(3, 3))

nens, nX = p.Xb.shape

ax = plt.subplot(1, 1, 1)
ax.scatter(p.iStorm_ens, p.jStorm_ens, s=3, color='.7')

g.plot_wind_contour(ax, p.ni, p.nj, p.Xt, 'black', 2)

cmap = [plt.cm.jet(x) for x in np.linspace(0, 1, p.nens_show)]
for n in range(p.nens_show):
  ax.scatter(p.iStorm_ens[n], p.jStorm_ens[n], s=40, color=[cmap[n][0:3]])
  # g.plot_wind_contour(ax, p.ni, p.nj, p.Xb[n, :], [cmap[n][0:3]], 2)

g.set_axis(ax, p.ni, p.nj)
# ax.tick_params(labelsize=15)

plt.savefig('ens.png', dpi=100)

