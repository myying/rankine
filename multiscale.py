import numpy as np
from rankine_vortex import *
from obs_def import *

###Pyramid method for spatial fields
###resolution level 1 correspond to the original field ni*nj, top-level has 2*2 grid points 2**lev=ni
##generate location coordinates (i, j, k) for each grid point
def get_loc(ni, nj, nv, lev):
    intv = 2 ** (lev-1)
    ii, jj, kk = np.mgrid[0:ni:intv, 0:nj:intv, 0:nv]
    Xloc = np.zeros((3, int(ni/intv), int(nj/intv), nv))
    Xloc[0, :, :, :] = ii
    Xloc[1, :, :, :] = jj
    Xloc[2, :, :, :] = kk
    return Xloc

def get_clev(wn):
    lev = 5 - np.log(wn)/np.log(2)
    return lev

def get_nobs_thin(wn, ni, nj, nobs):
    clev = get_clev(wn)
    nobs1 = np.minimum(nobs, ni*nj/(2**(clev-1)))
    return int(nobs1)

##scale decomposition
def lowpass_resp(Kh, k1, k2):
    r = np.zeros(Kh.shape)
    r[np.where(Kh<k1)] = 1.0
    r[np.where(Kh>k2)] = 0.0
    ind = np.where(np.logical_and(Kh>=k1, Kh<=k2))
    r[ind] = np.cos((Kh[ind] - k1)*(0.5*np.pi/(k2 - k1)))**2
    return r
def get_scale(x, kr, s):
    xk = grid2spec(x)
    xkout = xk.copy()
    ns = len(kr)
    if ns > 1:
        kx, ky = get_wn(x)
        Kh = np.sqrt(kx**2 + ky**2)
        if s == 0:
            xkout = xk * lowpass_resp(Kh, kr[s], kr[s+1])
        if s == ns-1:
            xkout = xk * (1 - lowpass_resp(Kh, kr[s-1], kr[s]))
        if s > 0 and s < ns-1:
            xkout = xk * (lowpass_resp(Kh, kr[s], kr[s+1]) - lowpass_resp(Kh, kr[s-1], kr[s]))
    return spec2grid(xkout)

def obs_get_scale(ni, nj, nv, Y, Ymask, Yloc, kr, s):
    Ys = Y.copy()
    nobs = Y.size
    ns = len(kr)
    if ns > 1:
        x = np.zeros((ni, nj, nv))
        kx, ky = get_wn(x)
        Kh = np.sqrt(kx**2 + ky**2)
        if s == 0:
            Ys = obs_convol(ni, nj, Y, Ymask, Yloc, spec2grid(lowpass_resp(Kh, kr[s], kr[s+1]))*ni*nj*nv/nobs)
        if s == ns-1:
            Ys = Y - obs_convol(ni, nj, Y, Ymask, Yloc, spec2grid(lowpass_resp(Kh, kr[s-1], kr[s]))*ni*nj*nv/nobs)
        if s > 0 and s < ns-1:
            Ys = obs_convol(ni, nj, Y, Ymask, Yloc, spec2grid(lowpass_resp(Kh, kr[s], kr[s+1]))*ni*nj*nv/nobs)
            Ys -= obs_convol(ni, nj, Y, Ymask, Yloc, spec2grid(lowpass_resp(Kh, kr[s-1], kr[s]))*ni*nj*nv/nobs)
    return Ys

def obs_convol(ni, nj, Y, Ymask, Yloc, r):
    Yout = np.zeros(Y.shape)
    for n in range(Y.size):
        if Ymask[n] == 1:
            dloc = np.zeros(Yloc.shape)
            dloc[0, :] = np.minimum(np.abs(Yloc[0, :]-Yloc[0, n]),ni-np.abs(Yloc[0, :]-Yloc[0, n]))
            dloc[1, :] = np.minimum(np.abs(Yloc[1, :]-Yloc[1, n]),nj-np.abs(Yloc[1, :]-Yloc[1, n]))
            dloc[2, :] = Yloc[2, :]
            w = obs_interp2d(r, dloc)
            w[np.where(Yloc[2, :]!=Yloc[2, n])] = 0.0
            Yout[n] = np.sum(w*Y)
    return Yout



###define scale band wavenumbers given dimensions
def get_krange(ns):
    kmax = 16 #np.minimum(16, int(np.sqrt(nobs)/2))
    dk = kmax**(1/ns)
    krange = dk**(np.arange(ns)+1)
    return krange

def get_obs_err_scale(ni, nj, nv, nobs, krange, s):
    Y = np.zeros(nobs*nv)
    Ymask = np.zeros(nobs*nv)
    Y[0] = 1.0
    Ymask[0] = 1
    Yloc = gen_obs_loc(ni, nj, nv, nobs)
    Ys = obs_get_scale(ni, nj, nv, Y, Ymask, Yloc, krange, s)
    obs_err_scale = Ys[0]
    return obs_err_scale

def get_local_cutoff(ns):
    if ns==1:
        local_cutoff = (16,)
    if ns==2:
        local_cutoff = (16, 8)
    if ns==3:
        local_cutoff = (18, 12, 8)
    if ns==4:
        local_cutoff = (20, 16, 12, 8)
    if ns==5:
        local_cutoff = (22, 20, 16, 12, 8)
    if ns==6:
        local_cutoff = (24, 22, 20, 16, 12, 8)
    if ns==7:
        local_cutoff = (26, 22, 20, 18, 16, 12, 8)
    return local_cutoff

def get_local_dampen(ns):
    if ns==1:
        local_dampen = (1.0,)
    if ns==2:
        local_dampen = (0.8, 0.9)
    if ns==3:
        local_dampen = (0.7, 0.7, 0.8)
    if ns==4:
        local_dampen = (0.5, 0.5, 0.7, 0.8)
    if ns==5:
        local_dampen = (0.4, 0.4, 0.6, 0.7, 0.8)
    if ns==6:
        local_dampen = (0.4, 0.4, 0.5, 0.7, 0.8, 0.8)
    if ns==7:
        local_dampen = (0.3, 0.3, 0.4, 0.5, 0.7, 0.8, 0.8)
    return local_dampen

###
def get_equal_cost_nens(nens, ns):
    kr = get_krange(ns)
    factor = 0
    for wn in kr:
        factor += 1/2**(2*(get_clev(wn)-1))
    nens1 = int(nens/factor)
    return nens1

