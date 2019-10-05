#!/usr/bin/env python

import numpy as np
import rankine_vortex as rv
import graphics as g
import config as p
import matplotlib.pyplot as plt

plt.switch_backend('Agg')

for x_in in range(41):
  for y_in in range(41):
    iout2 = np.array([x_in])
    jout2 = np.array([y_in])
    iout1 = np.array([17])
    jout1 = np.array([29])

    plt.figure(figsize=(3, 3))

    ###scatter plot
    ax = plt.subplot(1, 1, 1)
    Hout = rv.obs_operator(p.iX, p.jX, p.nv, iout1, jout1, p.iSite, p.jSite)
    err1 = np.dot(Hout, p.Xb.T) - np.dot(Hout, p.Xt)
    Hout = rv.obs_operator(p.iX, p.jX, p.nv, iout2, jout2, p.iSite, p.jSite)
    err2 = np.dot(Hout, p.Xb.T) - np.dot(Hout, p.Xt)
    ax.scatter(err1[0, :], err2[0, :], s=3, c='.7')
    cmap = [plt.cm.jet(x) for x in np.linspace(0, 1, p.nens_show)]
    for n in range(p.nens_show):
      ax.scatter(err1[0, n], err2[0, n], s=40, c=[cmap[n][0:3]])
    ax.set_xlim(-30, 30)
    ax.set_ylim(-30, 30)
    ax.tick_params(labelsize=8)

    print(x_in, y_in)
    plt.savefig('/glade/work/mying/visual/rankine/loc_sprd_{}'.format(p.loc_sprd)+'/error_scatter/{}_{}.png'.format(x_in, y_in), dpi=100)
    plt.close()

