import numpy as np
import rankine_vortex as rv
import localize

def EnSRF(ni, nj, nv, Xb, iX, jX, H, iObs, jObs, obs, obserr, localize_cutoff):
  nens, nX = Xb.shape
  nobs = obs.size
  X = Xb.copy()

  for p in range(nobs):
    h = H[p, :]
    yo = obs[p]
    hX = np.matmul(h, X.T)
    Xm = np.mean(X, axis=0)
    Xp = X - np.tile(Xm, (nens, 1))
    hXm = np.mean(hX)
    hXp = hX - hXm

    innov = yo - hXm
    varo = obserr**2
    varb = np.sum(hXp**2) / (nens - 1)
    cov = np.sum(Xp * np.tile(hXp, (nX, 1)).T, axis=0) / (nens - 1)

    dist = np.sqrt((iX - iObs[p])**2 + (jX - jObs[p])**2)
    loc = localize.GC(dist, localize_cutoff)
    loc1 = np.zeros((ni*nj*nv,))
    for v in range(nv):
      loc1[v*ni*nj:(v+1)*ni*nj] = loc
    gain = loc1 * cov / (varo + varb)
    srf = 1.0 / (1.0 + np.sqrt(varo / (varo + varb)))

    Xm = Xm + gain * innov
    for n in range(nens):
      Xp[n, :] = Xp[n, :] - srf * gain * hXp[n]
    X = Xp + np.tile(Xm, (nens, 1))

    print('{:-4d} ({:4.1f},{:4.1f}) yo={:6.2f} hxm={:6.2f} varb={:6.2f}'.format(p+1, iObs[p], jObs[p], obs[p], hXm, varb))

  return X

def RHF(ni, nj, nv, Xb, iX, jx, H, iObs, jObs, obs, obserr, localize_cutoff):
  nens, nX = Xb.shape
  nobs = obs.size
  X = Xb.copy()

  for p in range(nobs):
    h = H[p, :]
    yo = obs[p]
    hX = np.matmul(h, X.T)
    Xm = np.mean(X, axis=0)
    Xp = X - np.tile(Xm, (nens, 1))
    hXm = np.mean(hX)
    hXp = hX - hXm

    innov = yo - hXm
    varo = obserr**2
    varb = np.sum(hXp**2) / (nens - 1)
    cov = np.sum(Xp * np.tile(hXp, (nX, 1)).T, axis=0) / (nens - 1)

    dist = np.sqrt((iX - iObs[p])**2 + (jX - jObs[p])**2)
    loc = localize.GC(dist, localize_cutoff)
    loc1 = np.zeros((ni*nj*nv,))
    for v in range(nv):
      loc1[v*ni*nj:(v+1)*ni*nj] = loc
    gain = loc1 * cov / (varo + varb)
    srf = 1.0 / (1.0 + np.sqrt(varo / (varo + varb)))

    Xm = Xm + gain * innov
    for n in range(nens):
      Xp[n, :] = Xp[n, :] - srf * gain * hXp[n]
    X = Xp + np.tile(Xm, (nens, 1))

    print('{:-4d} ({:4.1f},{:4.1f}) yo={:6.2f} hxm={:6.2f} varb={:6.2f}'.format(p+1, iObs[p], jObs[p], obs[p], hXm, varb))

  return X


def LPF(ni, nj, nv, Xb, iX, jX, H, iObs, jObs, obs, obserr, localize_cutoff):
  nens, nX = Xb.shape
  nobs = obs.size
  X = Xb.copy()
  Neff = 1.1
  varo = obserr**2

  hxo = np.matmul(H, X.T)
  niter = 1

  hw = np.zeros((nobs, nens))
  for p in range(nobs):
    hw[p, :] = np.exp(-(obs[p] - hxo[p, :])**2 / (2*varo*niter))
    hw[p, :] = hw[p, :] / np.sum(hw[p, :])

  w = np.ones(X.shape) / nens
  wo = np.zeros(X.shape)
  w2 = np.ones((nens,)) / nens

  for p in range(nobs):
    h = H[p, :]
    ob_i = iObs[p]
    ob_j = jObs[p]
    yo = obs[p]
    hXm = np.mean(hxo[p, :])
    innov = yo - hXm
    varb = np.sum((hxo[p, :] - hXm)**2) / (nens-1)
    if varb == 0:
      print('variance collapsed!')
      continue

    dist = np.sqrt((iX - iObs[p])**2 + (jX - jObs[p])**2)
    loc = localize.GC(dist, localize_cutoff)
    loc1 = np.zeros((ni*nj*nv,))
    for v in range(nv):
      loc1[v*ni*nj:(v+1)*ni*nj] = loc

    print('{:-4d} ({:4.1f},{:4.1f}) yo={:6.2f} hxm={:6.2f} varb={:6.2f}'.format(p+1, iObs[p], jObs[p], obs[p], hXm, varb))

  return X


def find_beta():
  beta = 1.0
  return beta


def resample(x, w, nens):
  ind = x.argsort()
  cum_weight = np.zeros((nens+1,))
  cum_weight[1:] = np.cumsum(w[ind])
  for n in range(nens):
    frac = 1/(2*nens) + (n-1)/nens

