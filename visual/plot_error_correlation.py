#!/usr/bin/env python

import numpy as np
import rankine_vortex as rv
import graphics as g
import config as p
import matplotlib.pyplot as plt

plt.switch_backend('Agg')

for x_in in range(41):
  for y_in in range(41):
    iout = np.array([x_in])
    jout = np.array([y_in])

    plt.figure(figsize=(3, 3))
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
    g.plot_wind_contour(ax, p.ni, p.nj, p.Xt, 'black', 2)
    ax.plot(iout, jout, 'k+', markersize=10)
    ax.tick_params(labelsize=15)

    print(x_in, y_in)
    plt.savefig('/glade/work/mying/visual/rankine/loc_sprd_{}'.format(p.loc_sprd)+'/error_correlation/{}_{}.png'.format(x_in, y_in), dpi=100)
    plt.close()

