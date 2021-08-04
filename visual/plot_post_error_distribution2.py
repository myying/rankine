#!/usr/bin/env python

import numpy as np
import rankine_vortex as rv
import graphics as g
import config as p
import matplotlib.pyplot as plt
import data_assimilation as DA

plt.switch_backend('Agg')

x_in = 17
y_in = 29
iout1 = np.array([x_in])
jout1 = np.array([y_in])
H = rv.obs_operator(p.iX, p.jX, p.nv, iout1, jout1, p.iSite, p.jSite)
obs = np.matmul(H, p.Xt) + np.random.normal(0.0, p.obserr)
# Xa = DA.EnSRF(p.ni, p.nj, p.nv, p.Xb, p.iX, p.jX, H, iout1, jout1, obs, p.obserr, p.localize_cutoff)
Xa = DA.RHF(p.ni, p.nj, p.nv, p.Xb, p.iX, p.jX, H, iout1, jout1, obs, p.obserr, p.localize_cutoff)

for x_in in range(41):
  for y_in in range(41):
    iout2 = np.array([x_in])
    jout2 = np.array([y_in])

    plt.figure(figsize=(2, 3))

    ###plot histogram
    ax = plt.subplot(1, 1, 1)
    Hout = rv.obs_operator(p.iX, p.jX, p.nv, iout2, jout2, p.iSite, p.jSite)
    prior_err = np.dot(Hout, p.Xb.T) - np.dot(Hout, p.Xt)
    # err_mean = np.mean(prior_err)
    # err_std = np.std(prior_err)
    ii = np.arange(-50, 50, 1)
    # jj = np.exp(-0.5*(ii-err_mean)**2/ err_std**2) / np.sqrt(2*np.pi) / err_std
    jj0 = g.hist_normal(ii, prior_err[0, :])
    ax.plot(jj0, ii, 'k', linewidth=2, label='Sample')
    # ax.plot(jj, ii, 'r:', linewidth=1, label='Gaussian')
    # ax.legend(fontsize=12, loc=1)

    post_err = np.dot(Hout, Xa.T) - np.dot(Hout, p.Xt)
    ii = np.arange(-50, 50, 1)
    jj0 = g.hist_normal(ii, post_err[0, :])
    ax.plot(jj0, ii, 'r', linewidth=2, label='Sample')

    cmap = [plt.cm.jet(x) for x in np.linspace(0, 1, p.nens_show)]
    for n in range(p.nens_show):
      ax.scatter(0.1+0.01*n, prior_err[0, n], s=20, c=[cmap[n][0:3]])
      ax.scatter(0.1+0.01*n, post_err[0, n], s=20, c=[cmap[n][0:3]], marker='s')
      ax.plot([0.1+0.01*n, 0.1+0.01*n], [prior_err[0, n], post_err[0, n]], 'k-', linewidth=0.5)

    ax.set_xlim(-0.05, 0.3)
    ax.set_ylim(-30, 30)
    ax.tick_params(labelsize=8)

    print(x_in, y_in)
    plt.savefig('/glade/work/mying/visual/rankine/loc_sprd_{}'.format(p.loc_sprd)+'/error_distribution2/{}_{}.png'.format(x_in, y_in), dpi=100)
    plt.close()

