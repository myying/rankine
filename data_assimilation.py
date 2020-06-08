import numpy as np
import localize
import rankine_vortex as rv

def EnSRF(ni, nj, nv, Xb, iX, jX, H, iObs, jObs, vObs, obs, obserr, local_cutoff):
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
    dist = np.zeros(nX)
    for v in range(nv):
      dist[v*ni*nj:(v+1)*ni*nj] = np.sqrt((iX - iObs[p])**2 + (jX - jObs[p])**2)
    loc = localize.GC(dist, local_cutoff)
    cov = np.sum(Xp * np.tile(hXp, (nX, 1)).T, axis=0) / (nens - 1)
    gain = loc * cov / (varo + varb)
    srf = 1.0 / (1.0 + np.sqrt(varo / (varo + varb)))
    Xm = Xm + gain * innov
    for n in range(nens):
      Xp[n, :] = Xp[n, :] - srf * gain * hXp[n]
    X = Xp + np.tile(Xm, (nens, 1))
    print('{:-4d} ({:4.1f},{:4.1f}) yo={:6.2f} hxm={:6.2f} varb={:6.2f}'.format(p+1, iObs[p], jObs[p], obs[p], hXm, varb))
  return X

def PF(ni, nj, nv, Xb, iX, jX, H, iObs, jObs, vObs, obs, obserr, local_cutoff):
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
    dist = np.zeros(nX)
    for v in range(nv):
      dist[v*ni*nj:(v+1)*ni*nj] = np.sqrt((iX - iObs[p])**2 + (jX - jObs[p])**2)
    loc = localize.GC(dist, local_cutoff)
    cov = np.sum(Xp * np.tile(hXp, (nX, 1)).T, axis=0) / (nens - 1)
    gain = loc * cov / (varo + varb)
    srf = 1.0 / (1.0 + np.sqrt(varo / (varo + varb)))
    Xm = Xm + gain * innov
    for n in range(nens):
      Xp[n, :] = Xp[n, :] - srf * gain * hXp[n]
    X = Xp + np.tile(Xm, (nens, 1))
    print('{:-4d} ({:4.1f},{:4.1f}) yo={:6.2f} hxm={:6.2f} varb={:6.2f}'.format(p+1, iObs[p], jObs[p], obs[p], hXm, varb))
  return X
