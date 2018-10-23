import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset


def plot_wind_contour(ax, u, v, lcolor, lwidth):
  wspd = np.sqrt(u**2 + v**2)
  ni, nj = u.shape
  x, y = np.mgrid[0:ni, 0:nj]
  ax.contour(x, y, wspd, (15,), colors=lcolor, linewidths=lwidth)


def plot_obs(ax, obsi, obsj, obs):
  nobs = obs.size
  obsmax = max(obs)
  obsmin = min(obs)
  cmap = [plt.cm.jet(x) for x in np.linspace(0, 1, round(obsmax-obsmin)+1)]
  for i in range(nobs):
    ind = int(round(obs[i] - obsmin) + 1)
    plt.scatter(obsi[i], obsj[i], s=80, c=[cmap[ind-1][0:3]])


def plot_ens(ax, ni, nj, Xens, Xt):
  nens, nX = Xens.shape
  cmap = [plt.cm.jet(x) for x in np.linspace(0, 1, nens)]
  for n in range(nens):
    u = np.reshape(Xens[n, 0:ni*nj], (ni, nj))
    v = np.reshape(Xens[n, ni*nj:2*ni*nj], (ni, nj))
    plot_wind_contour(ax, u, v, [cmap[n][0:3]], 2)
  ut = np.reshape(Xt[0:ni*nj], (ni, nj))
  vt = np.reshape(Xt[ni*nj:2*ni*nj], (ni, nj))
  plot_wind_contour(ax, ut, vt, 'black', 4)


def set_axis(ax, ni, nj):
  ax.set_aspect('equal', 'box')
  ax.set_xlim(0, ni-1)
  ax.set_ylim(0, nj-1)
  ax.set_xticks(np.arange(0, ni, 20))
  ax.set_yticks(np.arange(0, nj, 20))


def output_ens(filename, ni, nj, Xens, Xtruth):
  import os
  nens, nX = Xens.shape
  if os.path.exists(filename):
    os.remove(filename)
  f = Dataset(filename, 'w', format='NETCDF4_CLASSIC')
  ii = f.createDimension('i', ni)
  jj = f.createDimension('j', nj)
  mm = f.createDimension('m', nens+2)
  dat = f.createVariable('u', np.float32, ('m', 'j', 'i'))
  for n in range(nens):
    dat[n, :, :] = np.reshape(Xens[n, 0:ni*nj], (ni, nj)).T
  dat[nens, :, :] = np.mean(dat[0:nens, :, :], axis=0)
  dat[nens+1, :, :] = np.reshape(Xtruth[0:ni*nj], (ni, nj)).T
  dat = f.createVariable('v', np.float32, ('m', 'j', 'i'))
  for n in range(nens):
    dat[n, :, :] = np.reshape(Xens[n, ni*nj:2*ni*nj], (ni, nj)).T
  dat[nens, :, :] = np.mean(dat[0:nens, :, :], axis=0)
  dat[nens+1, :, :] = np.reshape(Xtruth[ni*nj:2*ni*nj], (ni, nj)).T
  f.close()
