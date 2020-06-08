import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import rankine_vortex as rv

def plot_contour(ax, ni, nj, X, lcolor, lwidth):
  x, y = np.mgrid[0:ni, 0:nj]
  u = np.reshape(X[0:ni*nj], (ni, nj))
  v = np.reshape(X[ni*nj:2*ni*nj], (ni, nj))

  ##plot wind component
  # ax.contour(x, y, u, (-15, 15), colors=lcolor, linewidths=lwidth)

  ##plot wind spead
  # wspd = rv.uv2wspd(u, v)
  # ax.contour(x, y, wspd, (30,), colors=lcolor, linewidths=lwidth)

  ##plot wind component
  zeta = rv.uv2zeta(u, v)
  ax.contour(x, y, zeta, (0.5,), colors=lcolor, linewidths=lwidth)


def plot_obs(ax, obsi, obsj, obs):
  nobs = obs.size
  obsmax = 30
  obsmin = -30
  cmap = [plt.cm.jet(x) for x in np.linspace(0, 1, round(obsmax-obsmin)+1)]
  for i in range(nobs):
    ind = max(min(int(round(obs[i] - obsmin)), int(round(obsmax-obsmin))), 0)
    plt.scatter(obsi[i], obsj[i], s=80, c=[cmap[ind][0:3]])


def hist_normal(bins, sample):
  n = bins.size
  nsample = sample.size
  count = np.ones(nsample)
  dist = np.zeros(n)
  for i in range(n):
    if i == 0:
      v1 = -np.inf
      v2 = (bins[i]+bins[i+1])/2
    if i == n-1:
      v1 = (bins[i-1]+bins[i])/2
      v2 = np.inf
    if i > 0 and i < n-1:
      v1 = (bins[i-1]+bins[i])/2
      v2 = (bins[i]+bins[i+1])/2
    ind = np.where(np.logical_and(sample > v1, sample <= v2))
    dist[i] = np.sum(count[ind])
  dist = dist/nsample
  return dist

def sample_correlation(x1, x2):
  nens = x1.size
  x1_mean = np.mean(x1)
  x2_mean = np.mean(x2)
  x1p = x1 - x1_mean
  x2p = x2 - x2_mean
  cov = np.sum(x1p * x2p)
  x1_norm = np.sum(x1p ** 2)
  x2_norm = np.sum(x2p ** 2)
  corr = cov/np.sqrt(x1_norm * x2_norm)
  return corr

def smooth1d(x, smth):
  if smth > 0:
    x_smooth = np.zeros(x.shape)
    cw = 0.0
    for i in np.arange(-smth, smth, 1):
      w = np.exp(-i**2/(smth/2.0)**2)
      cw += w
      x_smooth += w * np.roll(x, i)
    x_smooth = x_smooth/cw
  else:
    x_smooth = x
  return x_smooth

def set_axis(ax, ni, nj):
  ax.set_aspect('equal', 'box')
  ax.set_xlim(0, ni-1)
  ax.set_ylim(0, nj-1)
  # ax.set_xticks(np.arange(0, ni, 20))
  # ax.set_yticks(np.arange(0, nj, 20))
  ax.set_xticks([])
  ax.set_yticks([])


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
