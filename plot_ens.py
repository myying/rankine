#!/usr/bin/env python

import numpy as np
import rankine_vortex as rv
import graphics as g
import config as p
import matplotlib.pyplot as plt

plt.switch_backend('Agg')
plt.figure(figsize=(5, 5))

ax = plt.subplot(1, 1, 1)
g.plot_ens(ax, p.ni, p.nj, p.Xb, p.Xt)
g.set_axis(ax, p.ni, p.nj)
# ax.plot(iout, jout, 'wo')
# ax.tick_params(labelsize=15)

plt.savefig('1.pdf')

