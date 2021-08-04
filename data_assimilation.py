import numpy as np
import matplotlib.pyplot as plt
from rankine_vortex import *
from multiscale import *

###observation network
def gen_obs_loc(ni, nj, nv, nobs):
    Yloc2 = np.zeros((2, nobs))
    Yloc2[0, :] = np.random.uniform(0, ni, nobs)
    Yloc2[1, :] = np.random.uniform(0, nj, nobs)
    Yloc = np.zeros((3, nobs*nv))
    for k in range(nv):
        Yloc[0:2, k::nv] = Yloc2
        Yloc[2, k::nv] = k
    return Yloc

def obs_interp2d(X, Yloc):
    dims = X.shape
    ni, nj = (dims[0], dims[1])
    io = Yloc[0, :]
    jo = Yloc[1, :]
    vo = Yloc[2, :].astype(int)
    io1 = np.floor(io).astype(int) % ni
    jo1 = np.floor(jo).astype(int) % nj
    io2 = np.floor(io+1).astype(int) % ni
    jo2 = np.floor(jo+1).astype(int) % nj
    di = io - np.floor(io)
    dj = jo - np.floor(jo)
    Y = (1-di)*(1-dj)*X[io1, jo1, vo] + di*(1-dj)*X[io2, jo1, vo] + (1-di)*dj*X[io1, jo2, vo] + di*dj*X[io2, jo2, vo]
    return Y

def plot_obs(ax, ni, nj, nv, Y, Ymask, Yloc):
    obsmin, obsmax = (-30, 30)
    cmap = [plt.cm.bwr(x) for x in np.linspace(0, 1, round(obsmax-obsmin)+1)]
    subset = np.where(np.logical_and(Yloc[2, :]==0, Ymask==1))
    color_ind = np.maximum(np.minimum(np.round(Y[subset]-obsmin), int(round(obsmax-obsmin))), 0).astype(int)
    ax.scatter(Yloc[0, subset], Yloc[1, subset], s=30, color=np.array(cmap)[color_ind, 0:3])
    ax.set_xlim([0, ni])
    ax.set_ylim([0, nj])
    return


##top-level wrapper for update at one analysis cycle:
def filter_update(Xb, Yo, Ymask, Yloc, filter_kind, obs_err_std, obs_thin, local_cutoff,
                  krange, run_alignment, smooth_obs=True, print_out=False):
    X = Xb.copy()
    ni, nj, nv, nens = Xb.shape
    ns = len(krange)

    for s in range(ns):
        print('   process scale component {}'.format(s+1))
        ##get scale component for prior state
        Xbs = get_scale(X, krange, s)
        Xloc = get_loc(ni, nj, nv, 1)
        ##get scale component for obs
        if smooth_obs:
            Yos = obs_get_scale(ni, nj, nv, Yo, Ymask, Yloc, krange, s)
        else:
            Yos = Yo.copy()
        ##get scale component for obs prior
        Ybs = np.zeros((Yo.size, nens))
        for m in range(nens):
            Ybm = obs_interp2d(X[:, :, :, m], Yloc)
            Ybm[np.where(Ymask==0)] = 0.0
            if smooth_obs:
                Ybs[:, m] = obs_get_scale(ni, nj, nv, Ybm, Ymask, Yloc, krange, s)
            else:
                Ybs[:, m] = Ybm.copy()
        if smooth_obs:
            obs_err_scale = get_obs_err_scale(ni, nj, nv, Yo.size, krange, s)
        else:
            obs_err_scale = 1.0

        if filter_kind=='EnSRF':
            Xas = EnSRF(Xbs, Xloc, Ybs, Yos, Ymask, Yloc,
                        obs_err_std[s]*obs_err_scale, obs_thin[s], local_cutoff[s], print_out)

        if s < ns-1 and run_alignment:
            print('      run alignment')
            us, vs = optical_flow(Xbs, Xas, nlevel=6-scales[s], w=500)
            Xbsw = warp(Xbs, -us, -vs)
            u = sharpen(us * 2**(scales[s]-1), scales[s], 1)
            v = sharpen(vs * 2**(scales[s]-1), scales[s], 1)
            X = warp(X, -u, -v)  ##displacement adjustment
            X += sharpen(Xas - Xbsw, scales[s], 1)  ##amplitude adjustment
        else:
            X += Xas - Xbs

    return X


##EnSRF filter update:
def EnSRF(Xb, Xloc, Yb, Yo, Ymask, Yloc, obs_err_std, obs_thin, local_cutoff, print_out):
    ni, nj, nv, nens = Xb.shape
    nobs = Yo.size
    X = Xb.copy()
    Y = Yb.copy()
    for p in range(0, nobs, obs_thin):    ##cycle through each observation
        if Ymask[p] == 1:
            Xm = np.mean(X, axis=3)
            Xp = X - np.repeat(np.expand_dims(Xm, 3), nens, axis=3)
            Ym = np.mean(Y, axis=1)
            Yp = Y - np.repeat(np.expand_dims(Ym, 1), nens, axis=1)
            yo = Yo[p]
            yb_ens = Y[p, :]
            yb_mean = np.mean(yb_ens)
            ybp = yb_ens - yb_mean
            innov = yo - yb_mean
            varo = obs_err_std**2    ##obs error variance
            varb = np.sum(ybp**2) / (nens - 1)    ##prior error variance
            srf = 1.0 / (1.0 + np.sqrt(varo / (varo + varb)))    ##square root modification
            if print_out: ##print out obs info
                print(('u', 'v')[int(Yloc[2, p])] + ' obs at ({:4.1f}, {:4.1f}) '.format(Yloc[0, p], Yloc[1, p]) +
                    'yo={:5.2f} yo_var={:5.2f}, yb_mean={:5.2f}, yb_var={:5.2f}'.format(yo, varo, yb_mean, varb))
            if varb>0 and np.abs(innov)!=0:
                ##localization
                dist = get_dist(ni, nj, Xloc[0, :, :, :], Xloc[1, :, :, :], Yloc[0, p], Yloc[1, p])
                C = local_GC(dist, local_cutoff)
                ##Kalman gain
                gain = C * np.sum(Xp * np.tile(ybp, (ni, nj, nv, 1)), axis=3) / (nens - 1) / (varo + varb)
                ##update mean and perturbations
                Xm = Xm + gain * innov
                for m in range(nens):
                    Xp[:, :, :, m] = Xp[:, :, :, m] - srf * gain * ybp[m]
                X = Xp + np.repeat(np.expand_dims(Xm, 3), nens, axis=3)

                ##update observation priors
                dist1 = get_dist(ni, nj, Yloc[0, :], Yloc[1, :], Yloc[0, p], Yloc[1, p])
                C1 = local_GC(dist1, local_cutoff)
                gain1 = C1 * np.sum(Yp * np.tile(ybp, (nobs, 1)), axis=1) / (nens - 1) / (varo + varb)
                Ym = Ym + gain1 * innov
                for m in range(nens):
                    Yp[:, m] = Yp[:, m] - srf * gain1 * ybp[m]
                Y = Yp + np.repeat(np.expand_dims(Ym, 1), nens, axis=1)
    return X


##particle filter
#def PF


def get_dist(ni, nj, ii, jj, io, jo):
    dist = np.sqrt(np.minimum(np.abs(ii-io),ni-np.abs(ii-io))**2+np.minimum(np.abs(jj-jo),nj-np.abs(jj-jo))**2)
    return dist

def local_GC(dist, cutoff):
    loc = np.zeros(dist.shape)
    if cutoff>0:
        r = dist / (cutoff / 2)
        loc1 = (((-0.25*r + 0.5)*r + 0.625)*r - 5.0/3.0) * r**2 + 1
        ind1 = np.where(dist<cutoff/2)
        loc[ind1] = loc1[ind1]
        r[np.where(r==0)] = 1e-10
        loc2 = ((((r/12.0 - 0.5)*r + 0.625)*r + 5.0/3.0)*r - 5.0)*r + 4 - 2.0/(3.0*r)
        ind2 = np.where(np.logical_and(dist>=cutoff/2, dist<cutoff))
        loc[ind2] = loc2[ind2]
    else:
        loc = np.ones(dist.shape)
    return loc


###optical flow algorithm for alignment step:
def optical_flow(x1in, x2in, nlevel, w=100):
    x1 = x1in.copy()
    x2 = x2in.copy()
    ni, nj, nv, nens = x1.shape
    u = np.zeros((ni, nj, nv, nens))
    v = np.zeros((ni, nj, nv, nens))
    ###pyramid levels
    for lev in range(nlevel, -1, -1):
        x1w = warp(x1, -u, -v)
        x1c = coarsen(smooth(x1w, 2**(lev-1)), 1, lev)
        x2c = coarsen(smooth(x2, 2**(lev-1)), 1, lev)
        niter = 1000
        xdx = 0.5*(deriv_x(x1c) + deriv_x(x2c))
        xdy = 0.5*(deriv_y(x1c) + deriv_y(x2c))  
        xdt = x2c - x1c
        ###compute incremental flow using iterative solver
        du = np.zeros(xdx.shape)
        dv = np.zeros(xdx.shape)
        for k in range(niter):
            ubar = laplacian(du) + du
            vbar = laplacian(dv) + dv
            du = ubar - xdx*(xdx*ubar + xdy*vbar + xdt)/(w + xdx**2 + xdy**2)
            dv = vbar - xdy*(xdx*ubar + xdy*vbar + xdt)/(w + xdx**2 + xdy**2)
        u += sharpen(du*2**lev, lev, 1)
        v += sharpen(dv*2**lev, lev, 1)
    return u, v

##some spatial operators
##coarsening resolution from lev1 to lev2 (lev1<lev2)
def coarsen(xi, lev1, lev2):
    x = xi.copy()
    if lev1 < lev2:
        for k in range(lev1, lev2):
            x1 = x[::2, ::2]
            x = x1
    return x

##sharpen resolution from lev1 to lev2, fill in grid points with linear interpolation
def sharpen(xi, lev1, lev2):
    x = xi.copy()
    if lev1 > lev2:
        for k in range(lev1, lev2, -1):
            dim = list(x.shape)
            dim[0] = dim[0]*2
            x1 = np.zeros(dim)
            x1[0:dim[0]:2, :] = x
            x1[1:dim[0]:2, :] = 0.5*(np.roll(x, -1, axis=0) + x)
            dim[1] = dim[1]*2
            x2 = np.zeros(dim)
            x2[:, 0:dim[1]:2] = x1
            x2[:, 1:dim[1]:2] = 0.5*(np.roll(x1, -1, axis=1) + x1)
            x = x2
    return x

def warp(x, u, v):
    xw = x.copy()
    ni, nj, nv, nens = x.shape
    ii, jj, vv, mm = np.mgrid[0:ni, 0:nj, 0:nv, 0:nens]
    xw = interp2d(x, ii+u, jj+v, vv, mm)
    return xw

def interp2d(x, io, jo, vv, mm):
    ni, nj, nv, nens = x.shape
    io1 = np.floor(io).astype(int) % ni
    jo1 = np.floor(jo).astype(int) % nj
    io2 = np.floor(io+1).astype(int) % ni
    jo2 = np.floor(jo+1).astype(int) % nj
    di = io - np.floor(io)
    dj = jo - np.floor(jo)
    xo = (1-di)*(1-dj)*x[io1, jo1, vv, mm] + di*(1-dj)*x[io2, jo1, vv, mm] + (1-di)*dj*x[io1, jo2, vv, mm] + di*dj*x[io2, jo2, vv, mm]
    return xo

def deriv_x(f):
    return (np.roll(f, -1, axis=0) - np.roll(f, 1, axis=0)) / 2.0

def deriv_y(f):
    return (np.roll(f, -1, axis=1) - np.roll(f, 1, axis=1)) / 2.0

def laplacian(f):
    return ((np.roll(f, -1, axis=0) + np.roll(f, 1, axis=0) + np.roll(f, -1, axis=1) + np.roll(f, 1, axis=1))/6. + 
            (np.roll(np.roll(f, -1, axis=1), -1, axis=0) + np.roll(np.roll(f, -1, axis=1), 1, axis=0) + 
             np.roll(np.roll(f, 1, axis=1), -1, axis=0) + np.roll(np.roll(f, 1, axis=1), 1, axis=0))/12. - f)


###some performance metrics
def rmse(Xens, Xt):
    return np.sqrt(np.mean((np.mean(Xens, axis=3)-Xt)**2, axis=(0,1,2)))

def sprd(Xens):
    return np.sqrt(np.mean(np.std(Xens, axis=3)**2, axis=(0,1,2)))

def sawtooth(out_b, out_a):
    nt = out_b.size
    tt = np.zeros(nt*2)
    tt[0::2] = np.arange(0, nt)
    tt[1::2] = np.arange(0, nt)
    out_st = np.zeros(nt*2)
    out_st[0::2] = out_b
    out_st[1::2] = out_a
    return tt, out_st

def pwr_spec(f):
    ni, nj, nv = f.shape
    fk = np.zeros((ni, nj, nv), dtype=complex)
    nupi = int(np.ceil((ni+1)/2))
    nupj = int(np.ceil((nj+1)/2))
    nup = max(nupi, nupj)
    wni = fft_wn(ni)
    wnj = fft_wn(nj)
    kj, ki = np.meshgrid(wni, wnj)
    k2d = np.sqrt((ki*(nup/nupi))**2 + (kj*(nup/nupj))**2)
    for v in range(nv):
        fk[:, :, v] = np.fft.fft2(f[:, :, v])
    P = (np.abs(fk)/ni/nj)**2
    Ptot = np.mean(P, axis=2)  ##kinetic energy is averaged u, v variances
    ##sum wavenumber ki and kj within range k2d=w
    wn = np.arange(0.0, nup)
    pwr = np.zeros(nup)
    for w in range(nup):
        pwr[w] = np.sum(Ptot[np.where(np.ceil(k2d)==w)])
    return wn, pwr

def err_spec(Xens, Xt):
    wn, err_pwr = pwr_spec(np.mean(Xens, axis=3)-Xt)
    return wn, err_pwr

def sprd_spec(Xens):
    ni, nj, nv, nens = Xens.shape
    Xmean = np.mean(Xens, axis=3)
    nup = int(np.ceil((ni+1)/2))
    pwr_ens = np.zeros((nup, nens))
    for m in range(nens):
        wn, pwr_ens[:, m] = pwr_spec(Xens[:, :, :, m] - Xmean)
    sprd_pwr = np.sum(pwr_ens, axis=1) / (nens-1)
    return wn, sprd_pwr


