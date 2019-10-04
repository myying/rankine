#!/usr/bin/env python

import numpy as np
import rankine_vortex as rv
import graphics as g
import config as p
import matplotlib.pyplot as plt

plt.switch_backend('Agg')
plt.figure(figsize=(5, 5))

nens_show = 10

ax = plt.subplot(1, 1, 1)
cmap = [plt.cm.jet(x) for x in np.linspace(0, 1, nens_show)]
for n in range(nens_show):
  g.plot_wind_contour(ax, p.ni, p.nj, p.Xb[n, :], [cmap[n][0:3]], 2)
g.plot_wind_contour(ax, p.ni, p.nj, p.Xt, 'black', 4)
g.set_axis(ax, p.ni, p.nj)
# ax.plot(iout, jout, 'wo')
# ax.tick_params(labelsize=15)

plt.savefig('1.pdf')

