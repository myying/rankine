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

    plt.figure(figsize=(3, 3))

    ###scatter plot
    ax = plt.subplot(1, 1, 1)
    Hout = rv.obs_operator(p.iX, p.jX, p.nv, iout1, jout1, p.iSite, p.jSite)
    err1 = np.dot(Hout, p.Xb.T) - np.dot(Hout, p.Xt)
    err1_post = np.dot(Hout, Xa.T) - np.dot(Hout, p.Xt)
    Hout = rv.obs_operator(p.iX, p.jX, p.nv, iout2, jout2, p.iSite, p.jSite)
    err2 = np.dot(Hout, p.Xb.T) - np.dot(Hout, p.Xt)
    err2_post = np.dot(Hout, Xa.T) - np.dot(Hout, p.Xt)
    ax.scatter(err1[0, :], err2[0, :], s=3, c='.7')
    ax.scatter(err1_post[0, :], err2_post[0, :], s=3, c='.4')

    cmap = [plt.cm.jet(x) for x in np.linspace(0, 1, p.nens_show)]
    for n in range(p.nens_show):
      ax.scatter(err1[0, n], err2[0, n], s=20, c=[cmap[n][0:3]])
      ax.scatter(err1_post[0, n], err2_post[0, n], s=20, c=[cmap[n][0:3]], marker='s')
      ax.plot([err1[0, n], err1_post[0, n]], [err2[0, n], err2_post[0, n]], 'k-', linewidth=0.5)
    ax.set_xlim(-30, 30)
    ax.set_ylim(-30, 30)
    ax.tick_params(labelsize=8)

    print(x_in, y_in)
    plt.savefig('/glade/work/mying/visual/rankine/loc_sprd_{}'.format(p.loc_sprd)+'/error_scatter/{}_{}.png'.format(x_in, y_in), dpi=100)
    plt.close()

