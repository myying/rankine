import numpy as np

def make_coords(ni, nj):
  ii, jj = np.mgrid[0:ni, 0:nj]
  iX = np.reshape(ii, (ni*nj,))
  jX = np.reshape(jj, (ni*nj,))
  return iX, jX

####rankine wind profile
def make_state(ni, nj, nv, iStorm, jStorm, Rmw, Vmax, Vout):
  iX, jX = make_coords(ni, nj)
  dist = np.sqrt((iX - iStorm)**2 + (jX - jStorm)**2)
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

####vortex with gauss vorticity and background flow
def make_state1(ni, nj, nv, dx, iStorm, jStorm, Rmw):
  power_law = -3
  pk = lambda k: k**((power_law-1)/2)
  zeta = 1e-4*gaussian_random_field(pk, ni)
  ii, jj = np.mgrid[0:ni, 0:nj]
  zeta += 1e-3*np.exp(-((ii-iStorm)**2+(jj-jStorm)**2)/(2.0*Rmw**2))
  u, v = zeta2uv(zeta, dx)
  X = np.zeros((ni*nj*nv,))
  X[0:ni*nj] += np.reshape(u, (ni*nj,))
  X[ni*nj:2*ni*nj] += np.reshape(v, (ni*nj,))
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

##model forward step, RK4 CS
def advance_time(ni, nj, X, dx, nt, dt):
  u, v = X2uv(ni, nj, X)
  zeta = uv2zeta(u, v, dx)
  u1, v1, zeta1 = u.copy(), v.copy(), zeta.copy()
  u2, v2, zeta2 = u.copy(), v.copy(), zeta.copy()
  u3, v3, zeta3 = u.copy(), v.copy(), zeta.copy()
  for n in range(nt):
    rhs1 = -(u*deriv_x(zeta, dx)+v*deriv_y(zeta, dx))
    zeta1 = zeta + 0.5*dt*rhs1
    u1, v1 = zeta2uv(zeta1, dx)
    rhs2 = -(u1*deriv_x(zeta1, dx)+v1*deriv_y(zeta1, dx))
    zeta2 = zeta + 0.5*dt*rhs2
    u2, v2 = zeta2uv(zeta2, dx)
    rhs3 = -(u2*deriv_x(zeta2, dx)+v2*deriv_y(zeta2, dx))
    zeta3 = zeta + dt*rhs3
    u3, v3 = zeta2uv(zeta3, dx)
    rhs4 = -(u3*deriv_x(zeta3, dx)+v3*deriv_y(zeta3, dx))
    zeta = zeta + dt*(rhs1/6.0+rhs2/3.0+rhs3/3.0+rhs4/6.0)
    u, v = zeta2uv(zeta, dx)
  Xt = uv2X(u, v)
  return Xt

def deriv_x(f, dx):
  return (np.roll(f, -1, axis=0) - np.roll(f, 1, axis=0))/(2.0*dx)

def deriv_y(f, dx):
  return (np.roll(f, -1, axis=1) - np.roll(f, 1, axis=1))/(2.0*dx)

###observation space
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

def uv2wspd(u, v):
  wspd = np.sqrt(u**2 + v**2)
  return wspd

def psi2uv(psi, dx):
  u = -(np.roll(psi, -1, axis=1) - psi)/dx
  v = (np.roll(psi, -1, axis=0) - psi)/dx
  return u, v

def uv2zeta(u, v, dx):
  zeta = (v - np.roll(v, 1, axis=0) - u + np.roll(u, 1, axis=1))/dx
  return zeta

def psi2zeta(psi, dx):
  zeta = ((np.roll(psi, -1, axis=0) + np.roll(psi, 1, axis=0) + np.roll(psi, -1, axis=1) + np.roll(psi, 1, axis=1)) - 4.0*psi)/(dx**2)
  return zeta

def zeta2psi(zeta, dx):
  psi = np.zeros(zeta.shape)
  niter = 3000
  for i in range(niter):
    psi = ((np.roll(psi, -1, axis=0) + np.roll(psi, 1, axis=0) + np.roll(psi, -1, axis=1) + np.roll(psi, 1, axis=1)) - zeta*(dx**2))/4.0
  return psi

def zeta2uv(zeta, dx):
  psi = zeta2psi(zeta, dx)
  u, v = psi2uv(psi, dx)
  return u, v


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

##random field
def generate_fft_index(n):
  nup = int(n/2)
  wn = np.concatenate((np.arange(0, nup), np.arange(-nup, 0)))
  return wn

def gaussian_random_field(pk, n):
  wn = generate_fft_index(n)
  kx, ky = np.meshgrid(wn, wn)
  k2d = np.sqrt(kx**2 + ky**2)
  k2d[np.where(k2d==0.0)] = 1e-10
  noise = np.fft.fft2(np.random.normal(0, 1, (n, n)))
  amplitude = pk(k2d)
  amplitude[np.where(k2d==1e-10)] = 0.0
  noise1 = np.real(np.fft.ifft2(noise * amplitude))
  return (noise1 - np.mean(noise1))/np.std(noise1)

