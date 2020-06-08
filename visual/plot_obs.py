#!/usr/bin/env python

import numpy as np
import rankine_vortex as rv
import graphics as g
import config as p
import matplotlib.pyplot as plt

plt.switch_backend('Agg')
plt.figure(figsize=(5, 5))

# ###plot obs
ax = plt.subplot(1, 1, 1)
g.plot_obs(ax, p.iObs, p.jObs, p.obs)
g.set_axis(ax, p.ni, p.nj)
g.plot_wind_contour(ax, p.ni, p.nj, p.Xt, 'black', 4)
ax.tick_params(labelsize=15)

plt.savefig('1.pdf')
