#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from rankine_vortex import *
from obs_def import *
from data_assimilation import *
from multiscale import *
from config import *

np.random.seed(0)

##truth
Xt = gen_vortex(ni, nj, nv, Vmax, Rmw, 0)

##observation
obs_err = 3
obs_th = 45
obs_r = 5
Ymask = np.ones(2)
Yloc = np.zeros((3, 2))
Yloc[0, :] = 0.5*ni + obs_r*np.sin(obs_th*np.pi/180*np.ones(2))
Yloc[1, :] = 0.5*nj + obs_r*np.cos(obs_th*np.pi/180*np.ones(2))
Yloc[2, :] = np.array([0, 1])
Yt = obs_interp2d(Xt, Yloc)
Yo = Yt + obs_err * np.random.normal(0, 1, 2)

##prior ensemble
nens = 200
loc_sprd = 3
Xb = np.zeros((ni, nj, nv, nens))
for m in range(nens):
    Xb[:, :, :, m] = gen_vortex(ni, nj, nv, Vmax, Rmw, loc_sprd)

nc = 9  ##number of cases to save
Xc = np.zeros((ni, nj, nv, nens, nc))
Xc[:, :, :, :, 0] = Xb.copy()  ##prior

c = 1  ##EnSRF single scale
ns = 1
Xc[:, :, :, :, c] = filter_update(Xb, Yo, Ymask, Yloc, 'EnSRF', obs_err*np.ones(ns), np.zeros(ns), np.ones(ns), get_krange(ns), (1,), False)

c = 2  ##EnSRF_MS_5 (without alignment steps)
ns = 5
Xc[:, :, :, :, c] = filter_update(Xb, Yo, Ymask, Yloc, 'EnSRF', obs_err*np.ones(ns), np.zeros(ns), np.ones(ns), get_krange(ns), (1,), False)


##c=4-7 first four stages of EnSRF_MSA_5
##c=3: final EnSRF_MSA_5 analysis
uc = np.zeros((ni, nj, nv, nens, nc))  ##save displacement vector stages
vc = np.zeros((ni, nj, nv, nens, nc))
ns = 5
X = Xb.copy()
nobs = Yo.size
krange=get_krange(ns)
for s in range(ns):
    clev = int(get_clev(krange[s]))
    Xbs = coarsen(get_scale(X, krange, s), 1, clev)
    Xloc = get_loc(ni, nj, nv, clev)
    Xas = Xbs.copy()
    Yb = np.zeros((nobs, nens))
    for m in range(nens):
        Yb[:, m] = obs_forward(X[:, :, :, m], Yloc)
    Xas = EnSRF(Xas, Xloc, Yb, Yo, Ymask, Yloc, obs_err_std, 0, 1, False)
    if s < ns-1:
        us, vs = optical_flow(Xbs, Xas, nlevel=3, w=1)
        Xbsw = warp(Xbs, -us, -vs)
        u = refine(us * 2**(clev-1), clev, 1)
        v = refine(vs * 2**(clev-1), clev, 1)
        X = warp(X, -u, -v)  ##displacement adjustment
        X += refine(Xas - Xbsw, clev, 1)  ##additional amplitude adjustment
        Xc[:, :, :, :, s+4] = X.copy()
        uc[:, :, :, :, s+4] = u.copy()
        vc[:, :, :, :, s+4] = v.copy()
    else:
        X += refine(Xas - Xbs, clev, 1)
        Xc[:, :, :, :, 3] = X.copy()

# np.save('Xc.npy', Xc)
# np.save('uc.npy', uc)
# np.save('vc.npy', vc)
# Xc = np.load('Xc.npy')
# uc = np.load('uc.npy')
# vc = np.load('vc.npy')

##plot
plt.switch_backend('Agg')
plt.figure(figsize=(13, 6))
ii, jj = np.mgrid[0:ni, 0:nj]
iout = (ii-ni/2)*dx/1000
jout = (jj-nj/2)*dx/1000

m_show = 0 ##plot which member?
for c in range(8):
    ax = plt.subplot(2, 4, c+1)
    wspd = np.sqrt(Xc[:, :, 0, m_show, c]**2 + Xc[:, :, 1, m_show, c]**2)
    cax = ax.contourf(iout, jout, wspd, np.arange(0, 50, 5), cmap='Greys')
    # plt.colorbar(cax)
    ax.contour(iout, jout, wspd, (20,), colors='red', linewidths=2)
    wspd = np.sqrt(Xt[:, :, 0]**2 + Xt[:, :, 1]**2)
    ax.contour(iout, jout, wspd, (20,), colors='black', linewidths=2)
    if c>3:  ##plot displacement vectors for MSA stages
        sp = 5
        ax.quiver(iout[1::sp, 1::sp], jout[1::sp, 1::sp],
                  uc[1::sp, 1::sp, 0, m_show, c], vc[1::sp, 1::sp, 0, m_show, c],
                  scale=25, headwidth=10, headlength=10, headaxislength=8)
    ax.plot((Yloc[0, ::2]-ni/2)*dx/1000, (Yloc[1, ::2]-nj/2)*dx/1000, 'k+', markersize=10, markeredgewidth=2)
    ax.set_aspect('equal','box')
    ax.set_xlim(-135, 135)
    ax.set_ylim(-135, 135)
    ax.set_xticks(np.arange(-120, 121, 60))
    ax.set_yticks(np.arange(-120, 121, 60))


plt.savefig('out.pdf')
