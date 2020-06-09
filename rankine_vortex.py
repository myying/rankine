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

def X2uv(ni, nj, X):
  u = np.reshape(X[0:ni*nj], (ni, nj))
  v = np.reshape(X[ni*nj:2*ni*nj], (ni, nj))
  return u, v

def uv2X(u, v):
  ni, nj = u.shape
  X = np.zeros(ni*nj*2)
  X[0:ni*nj] = np.reshape(u, (ni*nj,))
  X[ni*nj:2*ni*nj] = np.reshape(v, (ni*nj,))
  return X

def location_operator(iX, jX, iObs, jObs):
  nobs = iObs.size
  nX = iX.size
  L = np.zeros((nobs, nX))
  for p in range(nobs):
    io = int(iObs[p])
    jo = int(jObs[p])
    di = iObs[p] - io
    dj = jObs[p] - jo
    L[p, np.where(np.logical_and(iX==io, jX==jo))] = (1-di)*(1-dj)
    L[p, np.where(np.logical_and(iX==io+1, jX==jo))] = di*(1-dj)
    L[p, np.where(np.logical_and(iX==io, jX==jo+1))] = (1-di)*dj
    L[p, np.where(np.logical_and(iX==io+1, jX==jo+1))] = di*dj
  return L


##direct observation of u,v
def obs_operator(iX, jX, nv, iObs, jObs, vObs):
  nobs = iObs.size
  nX = iX.size
  L = location_operator(iX, jX, iObs, jObs)
  H = np.zeros((nobs, nX*nv))
  for p in range(nobs):
    H[p, vObs[p]*nX:(vObs[p]+1)*nX] = L[p, :]
  return H

##radial velocity
# def obs_operator(iX, jX, nv, iObs, jObs, vObs):
#   iSite = 2
#   jSite = 2 ##location of radar site
#   nobs = iObs.size
#   nX = iX.size
#   L = location_operator(iX, jX, iObs, jObs)
#   H = np.zeros((nobs, nX*nv))
#   rX = np.sqrt((iX - iSite)**2 + (jX - jSite)**2)
#   rX[np.where(rX==0)] = 1e-10
#   for p in range(nobs):
#     l = L[p, :]
#     H[p, 0:nX] = l * (np.matmul(l, iX) - iSite) / np.matmul(l, rX)    # u*(i-iSite)/r
#     H[p, nX:2*nX] = l * (np.matmul(l, jX) - jSite) / np.matmul(l, rX) # v*(j-jSite)/r
#   return H

###some convertors
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

###get feature
def get_max_wind(u, v):
  wspd = uv2wspd(u, v)
  return np.max(wspd)

def get_center_ij(u, v):
  ni, nj = u.shape
  zeta = uv2zeta(u, v)
  zmax = -999
  imax = -1
  jmax = -1
  buff = 10
  for i in range(buff, ni-buff):
    for j in range(buff, nj-buff):
      z = np.sum(zeta[i-buff:i+buff, j-buff:j+buff])
      if z > zmax:
        zmax = z
        imax = i
        jmax = j
  return imax, jmax
