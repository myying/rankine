import numpy as np
import localize
import rankine_vortex as rv

def EnSRF(ni, nj, nv, Xb, Yb, iX, jX, H, iObs, jObs, vObs, obs, obserr, local_cutoff):
  nens, nX = Xb.shape
  nobs = obs.size
  X = Xb.copy()
  Y = Yb.copy()
  for p in range(nobs):
    Xm = np.mean(X, axis=0)
    Xp = X - np.tile(Xm, (nens, 1))
    Ym = np.mean(Y, axis=1)
    Yp = Y - np.tile(Ym, (nens, 1)).T
    yo = obs[p]
    hX = Y[p, :]
    hXm = np.mean(hX)
    hXp = hX - hXm
    varo = obserr**2
    varb = np.sum(hXp**2) / (nens - 1)
    srf = 1.0 / (1.0 + np.sqrt(varo / (varo + varb)))
    dist = localize.make_dist(ni, nj, nv, iX, jX, iObs[p], jObs[p], vObs[p])
    loc = localize.GC(dist, local_cutoff)
    gain = loc * np.sum(Xp * np.tile(hXp, (nX, 1)).T, axis=0) / (nens - 1) / (varo + varb)
    Xm = Xm + gain * (yo - hXm)
    for n in range(nens):
      Xp[n, :] = Xp[n, :] - srf * gain * hXp[n]
    X = Xp + np.tile(Xm, (nens, 1))
    gain1 = np.sum(Yp * np.tile(hXp, (nobs, 1)), axis=1) / (nens - 1) / (varo + varb)
    Ym = Ym + gain1 * (yo - hXm)
    for n in range(nens):
      Yp[:, n] = Yp[:, n] - srf * gain1 * hXp[n]
    Y = Yp + np.tile(Ym, (nens, 1)).T
    # print('{:-4d} ({:4.1f},{:4.1f}) yo={:6.2f} hxm={:6.2f} varb={:6.2f}'.format(p+1, iObs[p], jObs[p], obs[p], hXm, varb))
  return X

def PF(ni, nj, nv, Xb, Yb, iX, jX, H, iObs, jObs, vObs, obs, obserr, local_cutoff):
  nens, nX = Xb.shape
  nobs = obs.size
  X = Xb.copy()
  Y = Yb.copy()
  for p in range(nobs):
    yo = obs[p]
    hX = Y[p, :] #np.matmul(H[p, :], X.T)
    wb = np.zeros(nens)
    wb[:] = 1.0 / nens
    wa = wb * np.exp( -(yo - hX)**2 / (2*obserr**2))
    wa = wa / np.sum(wa)
    ind = np.zeros(nens)
    w_ind = np.argsort(wa)
    cw = np.cumsum(wa[w_ind])
    for m in range(nens):
      fac = float(m+1)/float(nens+1)
      for n in range(nens-1):
        if (fac>cw[n] and fac<=cw[n+1]):
          ind[m] = w_ind[n]
    # print(ind)
    Xtmp = X.copy()
    Ytmp = Y.copy()
    for m in range(nens):
      X[m, :] = Xtmp[int(ind[m]), :]
      Y[:, m] = Ytmp[:, int(ind[m])]
  return X

def optical_flow_HS(Im1, Im2, nlevel):
  ni, nj = Im1.shape
  u = np.zeros((ni, nj))
  v = np.zeros((ni, nj))
  for lev in range(nlevel, -1, -1):
    Im1warp = warp(Im1, -u, -v)
    Im1c = coarsen(Im1warp, lev)
    Im2c = coarsen(Im2, lev)
    niter = 20
    w = 100
    Ix = 0.5*(deriv_x(Im1c) + deriv_x(Im2c))
    Iy = 0.5*(deriv_y(Im1c) + deriv_y(Im2c))
    It = Im2c - Im1c
    du = np.zeros(Ix.shape)
    dv = np.zeros(Ix.shape)
    for k in range(niter):
      du[0, :] = 0 ###boundary
      du[-1, :] = 0
      du[:, 0] = 0
      du[:, -1] = 0
      dv[0, :] = 0
      dv[-1, :] = 0
      dv[:, 0] = 0
      dv[:, -1] = 0
      ubar = laplacian(du) + du
      vbar = laplacian(dv) + dv
      du = ubar - Ix*(Ix*ubar + Iy*vbar + It)/(w + Ix**2 + Iy**2)
      dv = vbar - Iy*(Ix*ubar + Iy*vbar + It)/(w + Ix**2 + Iy**2)
    u += sharpen(du*2**lev, lev)
    v += sharpen(dv*2**lev, lev)
  return u, v

def warp(Im, u, v):
  warp_Im = Im.copy()
  ni, nj = Im.shape
  for i in range(ni):
    for j in range(nj):
      warp_Im[i, j] = interp2d(Im, (i+u[i, j], j+v[i, j]))
  return warp_Im

def coarsen(Im, level):
  for k in range(level):
    ni, nj = Im.shape
    Im1 = 0.25*(Im[0:ni:2, :][:, 0:nj:2] + Im[1:ni:2, :][:, 0:nj:2] + Im[0:ni:2, 1:nj:2] + Im[1:ni:2, 1:nj:2])
    Im = Im1
  return Im

def sharpen(Im, level):
  for k in range(level):
    ni, nj = Im.shape
    Im1 = np.zeros((ni*2, nj))
    Im1[0:ni*2:2, :] = Im
    Im1[1:ni*2:2, :] = 0.5*(np.roll(Im, -1, axis=0) + Im)
    Im2 = np.zeros((ni*2, nj*2))
    Im2[:, 0:nj*2:2] = Im1
    Im2[:, 1:nj*2:2] = 0.5*(np.roll(Im1, -1, axis=1) + Im1)
    Im = Im2
  return Im

def interp2d(x, loc):
  ni, nj = x.shape
  io = loc[0]
  jo = loc[1]
  io1 = int(np.floor(io)) % ni
  jo1 = int(np.floor(jo)) % nj
  io2 = int(np.floor(io+1)) % ni
  jo2 = int(np.floor(jo+1)) % nj
  di = io - np.floor(io)
  dj = jo - np.floor(jo)
  xo = (1-di)*(1-dj)*x[io1, jo1] + di*(1-dj)*x[io2, jo1] + (1-di)*dj*x[io1, jo2] + di*dj*x[io2, jo2]
  return xo

def deriv_x(f):
  fx = 0.5*(np.roll(f, -1, axis=0) - np.roll(f, 1, axis=0))
  return fx

def deriv_y(f):
  fy = 0.5*(np.roll(f, -1, axis=1) - np.roll(f, 1, axis=1))
  return fy

def laplacian(f):
  del2f = (np.roll(f, -1, axis=0) + np.roll(f, 1, axis=0) + np.roll(f, -1, axis=1) + np.roll(f, 1, axis=1))/6 + (np.roll(np.roll(f, -1, axis=1), -1, axis=0) + np.roll(np.roll(f, -1, axis=1), 1, axis=0) + np.roll(np.roll(f, 1, axis=1), -1, axis=0) + np.roll(np.roll(f, 1, axis=1), 1, axis=0))/12 - f
  return del2f

def get_coords(psik):
  nkx, nky = psik.shape
  kmax = nky-1
  kx_, ky_ = np.mgrid[-kmax:kmax+1, 0:kmax+1]
  return kx_, ky_

def spec2grid(fieldk):
  nkx, nky = fieldk.shape
  nx = nkx+1
  ny = 2*nky
  field = np.zeros((nx, ny))
  tmp = np.fft.ifftshift(fullspec(fieldk))
  field = nx*ny*np.real(np.fft.ifft2(tmp))
  return field

def grid2spec(field):
  nx, ny = field.shape
  nkx = nx-1
  nky = int(ny/2)
  fieldk = np.zeros((nkx, nky), dtype=complex)
  tmp = np.fft.fft2(field)/nx/ny
  fieldk = halfspec(np.fft.fftshift(tmp))
  return fieldk

def halfspec(sfield):
  n1, n2 = sfield.shape
  kmax = int(n1/2)-1
  hfield = sfield[1:n1, kmax+1:n2]
  return hfield

def fullspec(hfield):
  nkx, nky = hfield.shape
  kmax = nky-1
  hres = nkx+1
  sfield = np.zeros((hres, hres), dtype=complex)
  fup = hfield
  fup[kmax-1::-1, 0] = np.conjugate(fup[kmax+1:nkx, 0])
  fdn = np.conjugate(fup[::-1, nky-1:0:-1])
  sfield[1:hres, nky:hres] = fup
  sfield[1:hres, 1:nky] = fdn
  return sfield

def spec_bandpass(xk, krange, s):
  kx_, ky_ = get_coords(xk)
  Kh = np.sqrt(kx_**2 + ky_**2)
  xkout = xk.copy()
  if len(krange) > 1:
    r = scale_response(Kh, krange, s)
    xkout = xkout * r
  return xkout

def scale_response(Kh, krange, s):
  ns = len(krange)
  r = np.zeros(Kh.shape)
  center_k = krange[s]
  if s == 0:
    r[np.where(Kh<=center_k)] = 1.0
  else:
    left_k = krange[s-1]
    ind = np.where(np.logical_and(Kh>=left_k, Kh<=center_k))
    r[ind] = np.cos((Kh[ind] - center_k)*(0.5*np.pi/(left_k - center_k)))**2
  if s == ns-1:
    r[np.where(Kh>=center_k)] = 1.0
  else:
    right_k = krange[s+1]
    ind = np.where(np.logical_and(Kh>=center_k, Kh<=right_k))
    r[ind] = np.cos((Kh[ind] - center_k)*(0.5*np.pi/(right_k - center_k)))**2
  return r

def get_scale(ni, nj, nv, X, krange, s):
  Xs = X.copy()
  for v in range(nv):
    xv = np.reshape(X[v*ni*nj:(v+1)*ni*nj], (ni, nj))
    xk = grid2spec(xv)
    xk = spec_bandpass(xk, krange, s)
    xvs = spec2grid(xk)
    Xs[v*ni*nj:(v+1)*ni*nj] = np.reshape(xvs, (ni*nj,))
  return Xs

