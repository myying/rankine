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
    iout1 = np.array([23])
    jout1 = np.array([26])
    nens_show = 10

    plt.figure(figsize=(10, 5))
###plot histogram
    ax = plt.subplot(1, 2, 1)
    Hout = rv.obs_operator(p.iX, p.jX, p.nv, iout2, jout2, p.iSite, p.jSite)
    prior_err = np.dot(Hout, p.Xb.T) - np.dot(Hout, p.Xt)
    err_mean = np.mean(prior_err)
    err_std = np.std(prior_err)
    ii = np.arange(-50, 50, 1)
    jj = np.exp(-0.5*(ii-err_mean)**2/ err_std**2) / np.sqrt(2*np.pi) / err_std
    jj0 = g.hist_normal(ii, prior_err[0, :])
    ax.plot(jj0, ii, 'k', linewidth=4, label='Sample')
    ax.plot(jj, ii, 'r:', linewidth=2, label='Gaussian')
    ax.legend(fontsize=12, loc=1)
    ax.set_xlim(-0.05, 0.5)
    ax.set_ylim(-30, 50)
    ax.tick_params(labelsize=15)

###scatter plot
    ax = plt.subplot(1, 2, 2)
    Hout = rv.obs_operator(p.iX, p.jX, p.nv, iout1, jout1, p.iSite, p.jSite)
    err1 = np.dot(Hout, p.Xb.T) - np.dot(Hout, p.Xt)
    Hout = rv.obs_operator(p.iX, p.jX, p.nv, iout2, jout2, p.iSite, p.jSite)
    err2 = np.dot(Hout, p.Xb.T) - np.dot(Hout, p.Xt)
    ax.scatter(err1[0, :], err2[0, :], s=0.3, c='k')
    cmap = [plt.cm.jet(x) for x in np.linspace(0, 1, nens_show)]
    for n in range(nens_show):
      ax.scatter(err1[0, n], err2[0, n], s=10, c=[cmap[n][0:3]])
    ax.set_xlim(-30, 50)
    ax.set_ylim(-30, 50)
    ax.tick_params(labelsize=15)

    print(x_in, y_in)
    plt.savefig('/glade/work/mying/visual/rankine/loc_sprd_{}'.format(p.loc_sprd)+'/two_variable1/error_distribution2/{}_{}.png'.format(x_in, y_in), dpi=100)
    plt.close()

