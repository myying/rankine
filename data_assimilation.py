import numpy as np
import localize
import rankine_vortex as rv
import inflate

# def filter_update(ni, nj, nv, Xb, iX, jX, H, iObs, jObs, vObs, obs, obserr, local_cutoff, infb, krange, filter_kind):
#   nens, nX = Xb.shape
#   X = Xb.copy()
#   infa = infb.copy()
#   ns = len(krange)
#   local_factor = localize.local_ms_factor(ns)
#   ###inflation
#   # infa[:, :, 0] = update_inflation(ni, nj, nv, X, np.dot(H, X.T), iX, jX, H, iObs, jObs, vObs, obs, obserr, local_cutoff, infb[:, :, 0])
#   # Xm = np.mean(X, axis=0)
#   # for m in range(nens):
#     # X[m, :] = Xm + infa[:, 0, 0] * (X[m, :] - Xm)
#   ##scale decomposition
#   Xs = np.zeros((nens, nX, ns))
#   for s in range(ns):
#     for m in range(nens):
#       Xs[m, :, s] = get_scale(ni, nj, nv, X[m, :], krange, s)
#   ###scale loop
#   for s in range(ns):
#     X = np.sum(Xs, axis=2)
#     local = local_cutoff * local_factor[s]
#     sigma_o = obserr
#     ###inflation
#     infa[:, :, s] = update_inflation(ni, nj, nv, Xs[:, :, s], np.dot(H, X.T), iX, jX, H, iObs, jObs, vObs, obs, obserr, local_cutoff, infb[:, :, s])
#     Xsm = np.mean(Xs[:, :, s], axis=0)
#     for m in range(nens):
#       Xs[m, :, s] = Xsm + infa[:, 0, s] * (Xs[m, :, s] - Xsm)
#     ###run filter
#     Xsb = Xs[:, :, s].copy()
#     X = np.sum(Xs, axis=2)
#     if filter_kind == 'EnSRF':
#       Xsa = EnSRF(ni, nj, nv, Xsb, np.dot(H, X.T), iX, jX, H, iObs, jObs, vObs, obs, sigma_o, local)
#     if filter_kind == 'EnKF':
#       Xsa = EnKF(ni, nj, nv, Xsb, np.dot(H, X.T), iX, jX, H, iObs, jObs, vObs, obs, sigma_o, local)
#     if filter_kind == 'PF':
#       Xsa = PF(ni, nj, nv, Xsb, np.dot(H, X.T), iX, jX, H, iObs, jObs, vObs, obs, sigma_o, local)
#     ###alignment
#     # Xs[:, :, s] = Xsa
#     if s < ns-1:
#       for v in range(nv):
#         xb = np.zeros((ni, nj, nens))
#         xa = np.zeros((ni, nj, nens))
#         xv = np.zeros((ni, nj, nens))
#         for m in range(nens):
#           xb[:, :, m] = np.reshape(Xsb[m, v*ni*nj:(v+1)*ni*nj], (ni, nj))
#           xa[:, :, m] = np.reshape(Xsa[m, v*ni*nj:(v+1)*ni*nj], (ni, nj))
#         qu, qv = optical_flow_HS(xb, xa, 5)
#         for i in range(s+1, ns):
#           for m in range(nens):
#             xv[:, :, m] = np.reshape(Xs[m, v*ni*nj:(v+1)*ni*nj, s], (ni, nj))
#           xv = warp(xv, -qu, -qv)
#           for m in range(nens):
#             Xs[m, v*ni*nj:(v+1)*ni*nj, s] = np.reshape(xv[:, :, m], (ni*nj,))
#   X = np.sum(Xs, axis=2)
#   return X, infa

def filter_update(ni, nj, nv, Xb, iX, jX, H, iObs, jObs, vObs, obs, obserr, local_cutoff, infb, krange, filter_kind):
  nens, nX = Xb.shape
  X = Xb.copy()
  infa = infb.copy()
  ns = len(krange)
  local_factor = localize.local_ms_factor(ns)
  ###scale loop
  for s in range(ns):
    local = local_cutoff * local_factor[s]
    sigma_o = obserr
    ###get scale component
    Xsb = X.copy()
    for m in range(nens):
      Xsb[m, :] = get_scale(ni, nj, nv, X[m, :], krange, s)
    Xsa = Xsb.copy()
    ###inflation
    infa[:, :, s] = update_inflation(ni, nj, nv, Xsb, np.dot(H, X.T), iX, jX, H, iObs, jObs, vObs, obs, obserr, local_cutoff, infb[:, :, s])
    Xsbm = np.mean(Xsb, axis=0)
    Xsbi = Xsb.copy()
    for m in range(nens):
      Xsbi[m, :] = Xsbm + infa[:, 0, s] * (Xsb[m, :] - Xsbm)
    Xsb = Xsbi
    # X = X - Xsb + Xsbi
    Y = np.dot(H, X.T)
    ###run filter
    if filter_kind == 'EnSRF':
      Xsa = EnSRF(ni, nj, nv, Xsb, Y, iX, jX, H, iObs, jObs, vObs, obs, sigma_o, local)
    if filter_kind == 'EnKF':
      Xsa = EnKF(ni, nj, nv, Xsb, Y, iX, jX, H, iObs, jObs, vObs, obs, sigma_o, local)
    if filter_kind == 'PF':
      Xsa = PF(ni, nj, nv, Xsb, Y, iX, jX, H, iObs, jObs, vObs, obs, sigma_o, local)
    ###alignment
    if s < ns-1:
      for v in range(nv):
        xb = np.zeros((ni, nj, nens))
        xa = np.zeros((ni, nj, nens))
        xv = np.zeros((ni, nj, nens))
        for m in range(nens):
          xb[:, :, m] = np.reshape(Xsb[m, v*ni*nj:(v+1)*ni*nj], (ni, nj))
          xa[:, :, m] = np.reshape(Xsa[m, v*ni*nj:(v+1)*ni*nj], (ni, nj))
          xv[:, :, m] = np.reshape(X[m, v*ni*nj:(v+1)*ni*nj], (ni, nj))
        qu, qv = optical_flow_HS(xb, xa, 5)
        xv = warp(xv, -qu, -qv)
        xv += xa - warp(xb, -qu, -qv)
        for m in range(nens):
          X[m, v*ni*nj:(v+1)*ni*nj] = np.reshape(xv[:, :, m], (ni*nj,))
    else:
      X += Xsa - Xsb
  return X, infa

def update_inflation(ni, nj, nv, Xb, Yb, iX, jX, H, iObs, jObs, vObs, obs, obserr, local_cutoff, infb):
  nens, nX = Xb.shape
  nobs = obs.size
  X = Xb.copy()
  Y = Yb.copy()
  Xm = np.mean(X, axis=0)
  Xp = X - np.tile(Xm, (nens, 1))
  Ym = np.mean(Y, axis=1)
  Yp = Y - np.tile(Ym, (nens, 1)).T
  ##adaptive inflation
  inf = infb.copy()
  for p in range(nobs):
    yo = obs[p]
    hX = Y[p, :]
    hXm = np.mean(hX)
    hXp = hX - hXm
    varo = obserr**2  ##obs error variance
    varb = np.sum(hXp**2) / (nens - 1)  ##prior error variance
    dist = localize.make_dist(ni, nj, nv, iX, jX, iObs[p], jObs[p], vObs[p])
    loc = localize.GC(dist, local_cutoff)
    for i in range(nX):
      cov = np.sum(Xp[:, i] * hXp) / (nens - 1)
      var = np.sum(Xp[:, i]**2) / (nens - 1)
      if var<=0.0:
        corr = 0.0
      else:
        corr = loc[i] * cov / np.sqrt(var * varb)
      if (corr > 0):
        inf[i, 0], inf[i, 1] = inflate.adaptive_inflation(inf[i, 0], inf[i, 1], hXm, varb, nens, yo, varo, corr)
  return inf

def EnSRF(ni, nj, nv, Xb, Yb, iX, jX, H, iObs, jObs, vObs, obs, obserr, local_cutoff):
  nens, nX = Xb.shape
  nobs = obs.size
  X = Xb.copy()
  Y = Yb.copy()
  for p in range(nobs):  ##cycle through each observation
    Xm = np.mean(X, axis=0)
    Xp = X - np.tile(Xm, (nens, 1))
    Ym = np.mean(Y, axis=1)
    Yp = Y - np.tile(Ym, (nens, 1)).T
    yo = obs[p]
    hX = Y[p, :]
    hXm = np.mean(hX)
    hXp = hX - hXm
    varo = obserr**2  ##obs error variance
    varb = np.sum(hXp**2) / (nens - 1)  ##prior error variance
    srf = 1.0 / (1.0 + np.sqrt(varo / (varo + varb)))  ##square root modification
    ##localization
    dist = localize.make_dist(ni, nj, nv, iX, jX, iObs[p], jObs[p], vObs[p])
    loc = localize.GC(dist, local_cutoff)
    ##Kalman gain
    gain = loc * np.sum(Xp * np.tile(hXp, (nX, 1)).T, axis=0) / (nens - 1) / (varo + varb)
    ##update mean and perturbations
    Xm = Xm + gain * (yo - hXm)
    for n in range(nens):
      Xp[n, :] = Xp[n, :] - srf * gain * hXp[n]
    X = Xp + np.tile(Xm, (nens, 1))
    ##update observation priors
    gain1 = np.sum(Yp * np.tile(hXp, (nobs, 1)), axis=1) / (nens - 1) / (varo + varb)
    Ym = Ym + gain1 * (yo - hXm)
    for n in range(nens):
      Yp[:, n] = Yp[:, n] - srf * gain1 * hXp[n]
    Y = Yp + np.tile(Ym, (nens, 1)).T
    # print('{:-4d} ({:4.1f},{:4.1f}) yo={:6.2f} hxm={:6.2f} varb={:6.2f}'.format(p+1, iObs[p], jObs[p], obs[p], hXm, varb))
  return X

def EnKF(ni, nj, nv, Xb, Yb, iX, jX, H, iObs, jObs, vObs, obs, obserr, local_cutoff):
  nens, nX = Xb.shape
  nobs = obs.size
  X = Xb.copy()
  Y = Yb.copy()
  for p in range(nobs):
    Xm = np.mean(X, axis=0)
    Xp = X - np.tile(Xm, (nens, 1))
    Ym = np.mean(Y, axis=1)
    Yp = Y - np.tile(Ym, (nens, 1)).T
    yo = obs[p] + np.random.normal(0, obserr, (nens))
    hX = Y[p, :]
    hXm = np.mean(hX)
    hXp = hX - hXm
    varo = obserr**2
    varb = np.sum(hXp**2) / (nens - 1)
    dist = localize.make_dist(ni, nj, nv, iX, jX, iObs[p], jObs[p], vObs[p])
    loc = localize.GC(dist, local_cutoff)
    gain = loc * np.sum(Xp * np.tile(hXp, (nX, 1)).T, axis=0) / (nens - 1) / (varo + varb)
    for n in range(nens):
      X[n, :] = X[n, :] + gain * (yo[n] - hX[n])
    gain1 = np.sum(Yp * np.tile(hXp, (nobs, 1)), axis=1) / (nens - 1) / (varo + varb)
    for n in range(nens):
      Y[:, n] = Y[:, n] + gain1 * (yo[n] - hX[n])
  return X

def PF(ni, nj, nv, Xb, Yb, iX, jX, H, iObs, jObs, vObs, obs, obserr, local_cutoff):
  nens, nX = Xb.shape
  nobs = obs.size
  X = Xb.copy()
  Y = Yb.copy()
  w = np.zeros(nens)
  w[:] = 1.0 / nens
  for p in range(nobs):
    yo = obs[p]
    hX = Y[p, :]
    w = w * np.exp( -np.abs(yo - hX)**2 / (2*obserr**2))
  if(np.sum(w)==0.0):
    w[:] = 1.0 / nens
  else:
    w = w / np.sum(w)
  ind = np.zeros(nens)
  w_ind = np.argsort(w)
  cw = np.cumsum(w[w_ind])
  for m in range(nens-1):
    fac = float(m+1)/float(nens)
    for n in range(nens-1):
      if (fac>cw[n] and fac<=cw[n+1]):
        ind[m] = w_ind[n+1]
  ind[-1] = w_ind[-1]
  # print(ind)
  Xtmp = X.copy()
  Ytmp = Y.copy()
  for m in range(nens):
    X[m, :] = Xtmp[int(ind[m]), :]
    Y[:, m] = Ytmp[:, int(ind[m])]
  return X

def optical_flow_HS(Im1, Im2, nlevel):
  ni, nj, nens = Im1.shape
  u = np.zeros((ni, nj, nens))
  v = np.zeros((ni, nj, nens))
  for lev in range(nlevel, -1, -1):
    Im1warp = warp(Im1, -u, -v)
    Im1c = coarsen(Im1warp, lev)
    Im2c = coarsen(Im2, lev)
    niter = 100
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
  ni, nj, nens = Im.shape
  ii, jj, mm = np.mgrid[0:ni, 0:nj, 0:nens]
  warp_Im = interp2d(Im, ii+u, jj+v, mm)
  return warp_Im

def coarsen(Im, level):
  for k in range(level):
    ni, nj, nens = Im.shape
    Im1 = 0.25*(Im[0:ni:2, 0:nj:2, :] + Im[1:ni:2, 0:nj:2, :] + Im[0:ni:2, 1:nj:2, :] + Im[1:ni:2, 1:nj:2, :])
    Im = Im1
  return Im

def sharpen(Im, level):
  for k in range(level):
    ni, nj, nens = Im.shape
    Im1 = np.zeros((ni*2, nj, nens))
    Im1[0:ni*2:2, :, :] = Im
    Im1[1:ni*2:2, :, :] = 0.5*(np.roll(Im, -1, axis=0) + Im)
    Im2 = np.zeros((ni*2, nj*2, nens))
    Im2[:, 0:nj*2:2, :] = Im1
    Im2[:, 1:nj*2:2, :] = 0.5*(np.roll(Im1, -1, axis=1) + Im1)
    Im = Im2
  return Im

def interp2d(x, io, jo, mm):
  ni, nj, nens = x.shape
  io1 = np.floor(io).astype(int) % ni
  jo1 = np.floor(jo).astype(int) % nj
  io2 = np.floor(io+1).astype(int) % ni
  jo2 = np.floor(jo+1).astype(int) % nj
  di = io - np.floor(io)
  dj = jo - np.floor(jo)
  xo = (1-di)*(1-dj)*x[io1, jo1, mm] + di*(1-dj)*x[io2, jo1, mm] + (1-di)*dj*x[io1, jo2, mm] + di*dj*x[io2, jo2, mm]
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

