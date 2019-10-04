#!/usr/bin/env python

import numpy as np
import rankine_vortex as rv
import graphics as g
import config as p
import matplotlib.pyplot as plt
import sys

plt.switch_backend('Agg')
plt.figure(figsize=(5, 5))

x_in = int(sys.argv[1])
y_in = int(sys.argv[2])
iout = np.array([x_in])
jout = np.array([y_in])

ax = plt.subplot(1, 1, 1)
Hout = rv.obs_operator(p.iX, p.jX, p.nv, iout, jout, p.iSite, p.jSite)
x1 = np.dot(Hout, p.Xb.T)
corr_map = np.zeros((p.ni, p.nj))
for n in range(p.ni*p.nj):
  i = p.iX[n]
  j = p.jX[n]
  x2 = p.Xb[:, n]
  corr_map[i, j] = g.sample_correlation(x1, x2)
ii, jj = np.mgrid[0:p.ni, 0:p.nj]
ax.contourf(ii, jj, corr_map, np.arange(-1, 1.2, 0.1), cmap='bwr')
g.set_axis(ax, p.ni, p.nj)
g.plot_wind_contour(ax, p.ni, p.nj, p.Xt, 'black', 4)
ax.plot(iout, jout, 'k+', markersize=10)
ax.tick_params(labelsize=15)

print(x_in, y_in)
plt.savefig('/glade/work/mying/visual/rankine/error_correlation/{}_{}.png'.format(x_in, y_in), dpi=100)
# plt.savefig('1.pdf')

