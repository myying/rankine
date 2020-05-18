import numpy as np


def make_coords(ni, nj):
  ii, jj = np.mgrid[0:ni, 0:nj]
  iX = np.reshape(ii, (ni*nj,))
  jX = np.reshape(jj, (ni*nj,))
  return iX, jX


def make_state(ni, nj, nv, iStorm, jStorm, Rmw, Vmax, Vout):
  iX, jX = make_coords(ni, nj)
  dist = np.sqrt((iX - iStorm)**2 + (jX - jStorm)**2)

  ####wind profile
  wspd = np.zeros((ni*nj,))
  ind = np.where(dist <= Rmw)
  wspd[ind] = Vmax * dist[ind] / Rmw
  ind = np.where(dist > Rmw)
  wspd[ind] = (Vmax - Vout) * Rmw / dist[ind] * np.exp(-(dist[ind] - Rmw)**2/200) + Vout
  wspd[np.where(dist==0)] = 0

  dist[np.where(dist==0)] = 1e-10

  X = np.zeros((ni*nj*nv,))
  X[0:ni*nj] = -wspd * (jX - jStorm) / dist  # u component
  X[ni*nj:2*ni*nj] = wspd * (iX - iStorm) / dist  # v component
  return X


def location_operator(iX, jX, iObs, jObs):
  nobs = iObs.size
  nX = iX.size
  L = np.zeros((nobs, nX))
  for p in range(nobs):
    # L[p, :] = np.logical_and(iX==iObs[p], jX==jObs[p])
    io = int(iObs[p])
    jo = int(jObs[p])
    di = iObs[p] - io
    dj = jObs[p] - jo
    L[p, np.where(np.logical_and(iX==io, jX==jo))] = (1-di)*(1-dj)
    L[p, np.where(np.logical_and(iX==io+1, jX==jo))] = di*(1-dj)
    L[p, np.where(np.logical_and(iX==io, jX==jo+1))] = (1-di)*dj
    L[p, np.where(np.logical_and(iX==io+1, jX==jo+1))] = di*dj
  return L


def obs_operator(iX, jX, nv, iObs, jObs, iSite, jSite):
  nobs = iObs.size
  nX = iX.size
  L = location_operator(iX, jX, iObs, jObs)
  H = np.zeros((nobs, nX*nv))
  rX = np.sqrt((iX - iSite)**2 + (jX - jSite)**2)
  rX[np.where(rX==0)] = 1e-10
  for p in range(nobs):
    l = L[p, :]
    # H[p, 0:nX] = l * (np.matmul(l, iX) - iSite) / np.matmul(l, rX)    # u*(i-iSite)/r
    # H[p, nX:2*nX] = l * (np.matmul(l, jX) - jSite) / np.matmul(l, rX) # v*(j-jSite)/r
    H[p, 0:nX] = l
  return H


def uv2vr(u, v):
  sitex = 15
  sitey = 15
  ni, nj = u.shape
  x, y = np.mgrid[0:ni, 0:nj]
  r = np.sqrt((x - sitex)**2 + (y - sitey)**2)
  r[np.where(r==0)] = 1e-10
  vr = (u*(x - sitex) + v*(y - sitey)) / r
  return vr


def uv2zeta(u, v):
  zeta = np.zeros(u.shape)
  ni, nj = u.shape
  zeta[1:ni-1, 1:nj-1] = 0.5*(v[2:ni, 1:nj-1] - v[0:ni-2, 1:nj-1]) - 0.5*(u[1:ni-1, 2:nj] - u[1:ni-1, 0:nj-2])
  return zeta


def uv2wspd(u, v):
  wspd = np.sqrt(u**2 + v**2)
  return wspd
