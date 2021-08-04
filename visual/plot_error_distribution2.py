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

    plt.figure(figsize=(2, 3))

    ###plot histogram
    ax = plt.subplot(1, 1, 1)
    Hout = rv.obs_operator(p.iX, p.jX, p.nv, iout2, jout2, p.iSite, p.jSite)
    prior_err = np.dot(Hout, p.Xb.T) - np.dot(Hout, p.Xt)
    err_mean = np.mean(prior_err)
    err_std = np.std(prior_err)
    ii = np.arange(-50, 50, 1)
    jj = np.exp(-0.5*(ii-err_mean)**2/ err_std**2) / np.sqrt(2*np.pi) / err_std
    jj0 = g.hist_normal(ii, prior_err[0, :])
    ax.plot(jj0, ii, 'k', linewidth=2, label='Sample')
    ax.plot(jj, ii, 'r:', linewidth=1, label='Gaussian')
    # ax.legend(fontsize=12, loc=1)

    cmap = [plt.cm.jet(x) for x in np.linspace(0, 1, p.nens_show)]
    for n in range(p.nens_show):
      ax.scatter(0, prior_err[0, n], s=40, c=[cmap[n][0:3]])

    ax.set_xlim(-0.05, 0.3)
    ax.set_ylim(-30, 30)
    ax.tick_params(labelsize=8)

    print(x_in, y_in)
    plt.savefig('/glade/work/mying/visual/rankine/loc_sprd_{}'.format(p.loc_sprd)+'/error_distribution2/{}_{}.png'.format(x_in, y_in), dpi=100)
    plt.close()

