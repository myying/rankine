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


# ###plot obs
# # ax = plt.subplot(1, 2, 2)
# # g.plot_obs(ax, iObs, jObs, obs)
# # g.set_axis(ax, ni, nj)
# # g.plot_wind_contour(ax, ni, nj, Xt, 'black', 4)
# # ax.tick_params(labelsize=15)

###plot histogram
ax = plt.subplot(1, 1, 1)
Hout = rv.obs_operator(p.iX, p.jX, p.nv, iout, jout, p.iSite, p.jSite)
prior_err = np.dot(Hout, p.Xb.T) - np.dot(Hout, p.Xt)
err_mean = np.mean(prior_err)
err_std = np.std(prior_err)
ii = np.arange(-50, 50, 1)
jj = np.exp(-0.5*(ii-err_mean)**2/ err_std**2) / np.sqrt(2*np.pi) / err_std
jj0 = g.hist_normal(ii, prior_err[0, :])
ax.plot(ii, jj0, 'k', linewidth=4, label='Sample')
ax.plot(ii, jj, 'r:', linewidth=2, label='Gaussian')
ax.legend(fontsize=12, loc=1)
ax.set_xlim(-30, 50)
ax.set_ylim(-0.05, 0.5)
ax.tick_params(labelsize=15)

# ax = plt.subplot(2, 3, 5)
# Hout = rv.obs_operator(iX, jX, nv, iout+ioff, jout+joff, iSite, jSite)
# prior_err = np.dot(Hout, Xb.T) - np.dot(Hout, Xt)
# err_mean = np.mean(prior_err)
# err_std = np.std(prior_err)
# ii = np.arange(-50, 50, 1)
# jj = np.exp(-0.5*(ii-err_mean)**2/ err_std**2) / np.sqrt(2*np.pi) / err_std
# jj0 = g.hist_normal(ii, prior_err[0, :])
# ax.plot(jj0, ii, 'k', linewidth=4, label='Sample')
# ax.plot(jj, ii, 'r:', linewidth=2, label='Gaussian')
# ax.legend(fontsize=12, loc=1)
# ax.set_ylim(-30, 50)
# ax.tick_params(labelsize=15)


print(x_in, y_in)
plt.savefig('/glade/work/mying/visual/rankine/error_distribution/{}_{}.png'.format(x_in, y_in), dpi=100)

