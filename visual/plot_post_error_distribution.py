#!/usr/bin/env python

import numpy as np
import rankine_vortex as rv
import graphics as g
import config as p
import data_assimilation as DA
import matplotlib.pyplot as plt

plt.switch_backend('Agg')

x_in = 17
y_in = 29
iout = np.array([x_in])
jout = np.array([y_in])
H = rv.obs_operator(p.iX, p.jX, p.nv, iout, jout, p.iSite, p.jSite)
obs = np.matmul(H, p.Xt) + np.random.normal(0.0, p.obserr)
# Xa = DA.EnSRF(p.ni, p.nj, p.nv, p.Xb, p.iX, p.jX, H, iout, jout, obs, p.obserr, p.localize_cutoff)
Xa = DA.RHF(p.ni, p.nj, p.nv, p.Xb, p.iX, p.jX, H, iout, jout, obs, p.obserr, p.localize_cutoff)

plt.figure(figsize=(3, 2))

###plot histogram
ax = plt.subplot(1, 1, 1)
prior_err = np.dot(H, p.Xb.T) - np.dot(H, p.Xt)
# err_mean = np.mean(prior_err)
# err_std = np.std(prior_err)
ii = np.arange(-50, 50, 1)
# jj = np.exp(-0.5*(ii-err_mean)**2/ err_std**2) / np.sqrt(2*np.pi) / err_std
jj0 = g.hist_normal(ii, prior_err[0, :])
ax.plot(ii, jj0, 'k', linewidth=2, label='Sample')
# ax.plot(ii, jj, 'r:', linewidth=1, label='Gaussian')
# ax.legend(fontsize=12, loc=1)

post_err = np.dot(H, Xa.T) - np.dot(H, p.Xt)
ii = np.arange(-50, 50, 1)
jj0 = g.hist_normal(ii, post_err[0, :])
ax.plot(ii, jj0, 'r', linewidth=2, label='Sample')

cmap = [plt.cm.jet(x) for x in np.linspace(0, 1, p.nens_show)]
for n in range(p.nens_show):
  ax.scatter(prior_err[0, n], 0.1+0.01*n, s=20, color=[cmap[n][0:3]])
  ax.scatter(post_err[0, n], 0.1+0.01*n, s=20, color=[cmap[n][0:3]], marker='s')
  ax.plot([prior_err[0, n], post_err[0, n]], [0.1+0.01*n, 0.1+0.01*n], 'k-', linewidth=0.5)

ax.set_xlim(-30, 30)
ax.set_ylim(-0.05, 0.3)
ax.tick_params(labelsize=8)

plt.savefig('/glade/work/mying/visual/rankine/loc_sprd_{}'.format(p.loc_sprd)+'/error_distribution/{}_{}.png'.format(x_in, y_in), dpi=100)
# plt.savefig('1.pdf')
plt.close()

