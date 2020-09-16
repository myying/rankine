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
def make_vortex(ni, nj, nv, dx, iStorm, jStorm, Rmw):
  ii, jj = np.mgrid[0:ni, 0:nj]
  zeta = 1e-3*np.exp(-((ii-iStorm)**2+(jj-jStorm)**2)/(2.0*Rmw**2))
  u, v = zeta2uv(zeta, dx)
  X = np.zeros((ni*nj*nv,))
  X[0:ni*nj] += np.reshape(u, (ni*nj,))
  X[ni*nj:2*ni*nj] += np.reshape(v, (ni*nj,))
  return X

def make_background_flow(ni, nj, nv, dx, ampl):
  power_law = -3
  pk = lambda k: k**((power_law-1)/2)
  zeta = ampl * gaussian_random_field(pk, ni)
  u, v = zeta2uv(zeta, dx)
  ioff = 20
  joff = -10
  u = np.roll(np.roll(u, ioff, axis=0), joff, axis=1)
  v = np.roll(np.roll(v, ioff, axis=0), joff, axis=1)
  X = np.zeros((ni*nj*nv,))
  X[0:ni*nj] += np.reshape(u, (ni*nj,))
  X[ni*nj:2*ni*nj] += np.reshape(v, (ni*nj,))
  return X

def X2uv(ni, nj, X):
  if X.ndim==1:
      u = np.reshape(X[0:ni*nj], (ni, nj))
      v = np.reshape(X[ni*nj:2*ni*nj], (ni, nj))
  if X.ndim==2:
    nX, nens = X.shape
    u = np.zeros((ni, nj, nens))
    v = np.zeros((ni, nj, nens))
    for m in range(nens):
      u[:, :, m] = np.reshape(X[0:ni*nj, m], (ni, nj))
      v[:, :, m] = np.reshape(X[ni*nj:2*ni*nj, m], (ni, nj))
  return u, v

def uv2X(ni, nj, u, v):
  if u.ndim==2:
    X = np.zeros((ni*nj*2))
    X[0:ni*nj] = np.reshape(u, (ni*nj,))
    X[ni*nj:2*ni*nj] = np.reshape(v, (ni*nj,))
  if u.ndim==3:
    ni, nj, nens = u.shape
    X = np.zeros((ni*nj*2, nens))
    for m in range(nens):
      X[0:ni*nj, m] = np.reshape(u[:, :, m], (ni*nj,))
      X[ni*nj:2*ni*nj, m] = np.reshape(v[:, :, m], (ni*nj,))
  return X

##model forward step, RK4 CS
def advance_time(ni, nj, X, dx, nt, dt, gen_rate):
  u, v = X2uv(ni, nj, X)
  zeta = uv2zeta(u, v, dx)
  diss = 5e3
  gen=gen_rate*1e-5*wind_cutoff(np.max(uv2wspd(u,v)), 70)
  for n in range(nt):
    rhs1 = forcing(u, v, zeta, diss, gen, dx)
    zeta1 = zeta + 0.5*dt*rhs1
    rhs2 = forcing(u, v, zeta1, diss, gen, dx)
    zeta2 = zeta + 0.5*dt*rhs2
    rhs3 = forcing(u, v, zeta2, diss, gen, dx)
    zeta3 = zeta + dt*rhs3
    rhs4 = forcing(u, v, zeta3, diss, gen, dx)
    zeta = zeta + dt*(rhs1/6.0+rhs2/3.0+rhs3/3.0+rhs4/6.0)
    u, v = zeta2uv(zeta, dx)
  Xt = uv2X(ni, nj, u, v)
  return Xt

def wind_cutoff(wind, max_wind):
  buff = 10.0
  f = 0.0
  if (wind < max_wind-buff):
    f = 1.0
  if (wind >= max_wind-buff and wind < max_wind):
    f = (max_wind - wind) / buff
  return f

def forcing(u, v, zeta, diss, gen, dx):
  fzeta = -(u*deriv_x(zeta, dx)+v*deriv_y(zeta, dx)) + gen*zeta + diss*laplacian(zeta, dx)
  return fzeta

def deriv_x(f, dx):
  return (np.roll(f, -1, axis=0) - np.roll(f, 1, axis=0))/(2.0*dx)

def deriv_y(f, dx):
  return (np.roll(f, -1, axis=1) - np.roll(f, 1, axis=1))/(2.0*dx)

def laplacian(f, dx):
  return ((np.roll(f, -1, axis=0) + np.roll(f, 1, axis=0) + np.roll(f, -1, axis=1) + np.roll(f, 1, axis=1)) - 4.0*f)/(dx**2)

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
    if vObs[p] == 0 or vObs[p] == 1:
      H[p, int(vObs[p]*nX):int((vObs[p]+1)*nX)] = L[p, :]
    if vObs[p] == 99:
      ##radial velocity
      iSite = 62
      jSite = 62 ##location of radar site
      rX = np.sqrt((iX - iSite)**2 + (jX - jSite)**2)
      rX[np.where(rX==0)] = 1e-10
      H[p, 0:nX] = L * (np.dot(L, iX) - iSite) / np.dot(L, rX)    # u*(i-iSite)/r
      H[p, nX:2*nX] = L * (np.dot(L, jX) - jSite) / np.dot(L, rX) # v*(j-jSite)/r
  return H

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
  niter = 1000
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

def get_center_ij(u, v, dx):
  ni, nj = u.shape
  zeta = uv2zeta(u, v, dx)
  zmax = -999
  imax = -1
  jmax = -1
  buff = 3
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

