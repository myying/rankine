import numpy as np

##utility functions: convert between state and spectral spaces
##we only consider square domains in this case (ni=nj)
def grid2spec(f):
    ni, nj = (f.shape[0], f.shape[1])
    return np.fft.fft2(f, axes=(0, 1))

def spec2grid(fk):
    ni, nj = (fk.shape[0], fk.shape[1])
    return np.real(np.fft.ifft2(fk, axes=(0, 1)))

##generate wavenumber sequence for fft results
def fft_wn(n):
    nup = int(np.ceil((n+1)/2))
    if n%2 == 0:
        wn = np.concatenate((np.arange(0, nup), np.arange(2-nup, 0)))
    else:
        wn = np.concatenate((np.arange(0, nup), np.arange(1-nup, 0)))
    return wn

##generate meshgrid wavenumber for input field x
## the first two dimensions are horizontal (i, j)
def get_wn(x):
    dims = x.shape
    n = dims[0]
    wn = fft_wn(dims[0])
    wni = np.expand_dims(wn, 1)
    wni = np.repeat(wni, dims[1], axis=1)
    wnj = np.expand_dims(wn, 0)
    wnj = np.repeat(wnj, dims[0], axis=0)
    for d in range(2, len(dims)):  ##extra dimensions
        wni = np.expand_dims(wni, d)
        wni = np.repeat(wni, dims[d], axis=d)
        wnj = np.expand_dims(wnj, d)
        wnj = np.repeat(wnj, dims[d], axis=d)
    return wni, wnj

##scaled wavenumber k for pseudospectral method
def get_scaled_wn(x, dx):
    n = x.shape[0]
    wni, wnj = get_wn(x)
    ki = (2.*np.pi) * wni / (n*dx)
    kj = (2.*np.pi) * wnj / (n*dx)
    return ki, kj


##random realization of model initial condition
##amp: wind speed amplitude; power_law: wind field power spectrum slope
def gen_random_flow(ni, nj, nv, dx, amp, power_law):
    x = np.zeros((ni, nj, nv))
    ##generate random streamfunction for the wind
    psi = random_field(ni, power_law-2)
    ##convert to wind
    u = -(np.roll(psi, -1, axis=1) - np.roll(psi, 1, axis=1)) / (2.0*dx)
    v = (np.roll(psi, -1, axis=0) - np.roll(psi, 1, axis=0)) / (2.0*dx)
    ##normalize and scale to required amp
    u = amp * (u - np.mean(u)) / np.std(u)
    v = amp * (v - np.mean(v)) / np.std(v)
    x[:, :, 0] = u
    x[:, :, 1] = v
    return x

##rankine vortex. Vmax: maximum wind speed, Rmw: radius of max wind, loc_sprd: the spread in center location
def gen_vortex(ni, nj, nv, Vmax, Rmw, loc_sprd=0, loc_bias=0):
    x = np.zeros((ni, nj, nv))
    ii, jj = np.mgrid[0:ni, 0:nj]
    center_i = 0.5*ni + loc_bias + np.random.normal(0, loc_sprd)
    center_j = 0.5*nj + loc_bias + np.random.normal(0, loc_sprd)
    dist = np.sqrt((ii-center_i)**2 + (jj-center_j)**2)
    dist[np.where(dist==0)] = 1e-10  ##avoid divide by 0
    wspd = np.zeros(dist.shape)
    ind = np.where(dist <= Rmw)
    wspd[ind] = Vmax * dist[ind] / Rmw
    ind = np.where(dist > Rmw)
    wspd[ind] = Vmax * (Rmw / dist[ind])**1.5
    wspd[np.where(dist==0)] = 0
    x[:, :, 0] = -wspd * (jj - center_j) / dist  ##u component
    x[:, :, 1] = wspd * (ii - center_i) / dist   ##v component
    return x

def random_field(n, power_law):
    pk = lambda k: k**((power_law-1)/2)
    wn = fft_wn(n)
    kx, ky = np.meshgrid(wn, wn)
    k2d = np.sqrt(kx**2 + ky**2)
    k2d[np.where(k2d==0.0)] = 1e-10
    noise = np.fft.fft2(np.random.normal(0, 1, (n, n)))
    amplitude = pk(k2d)
    amplitude[np.where(k2d==1e-10)] = 0.0
    noise1 = np.real(np.fft.ifft2(noise * amplitude))
    return (noise1 - np.mean(noise1))/np.std(noise1)


##model forecast step:
def advance_time(X, dx, dt, smalldt, gen, diss):
    ##input wind components, convert to spectral space
    u = grid2spec(X[:, :, 0])
    v = grid2spec(X[:, :, 1])
    ##convert to zeta
    ki, kj = get_scaled_wn(u, dx)
    zeta = 1j * (ki*v - kj*u)
    k2 = ki**2 + kj**2
    k2[np.where(k2==0)] = 1  #avoid singularity in inversion
    ##run time loop:
    for n in range(int(dt/smalldt)):
        ##use rk4 numeric scheme to integrate forward in time:
        rhs1 = forcing(u, v, zeta, dx, gen, diss)
        zeta1 = zeta + 0.5*smalldt*rhs1
        rhs2 = forcing(u, v, zeta1, dx, gen, diss)
        zeta2 = zeta + 0.5*smalldt*rhs2
        rhs3 = forcing(u, v, zeta2, dx, gen, diss)
        zeta3 = zeta + smalldt*rhs3
        rhs4 = forcing(u, v, zeta3, dx, gen, diss)
        zeta = zeta + smalldt*(rhs1/6.0 + rhs2/3.0 + rhs3/3.0 + rhs4/6.0)
        ##inverse zeta to get u, v
        psi = -zeta / k2
        u = -1j * kj * psi
        v = 1j * ki * psi
    X1 = X.copy()
    X1[:, :, 0] = spec2grid(u)
    X1[:, :, 1] = spec2grid(v)
    return X1

def forcing(u, v, zeta, dx, gen, diss):
    ki, kj = get_scaled_wn(zeta, dx)
    ug = spec2grid(u)
    vg = spec2grid(v)
    ##advection term:
    f = -grid2spec(ug*spec2grid(1j*ki*zeta) + vg*spec2grid(1j*kj*zeta))
    ##generation term:
    vmax = np.max(np.sqrt(ug**2+vg**2))
    if vmax > 75:  ##cut off generation if vortex intensity exceeds limit
        gen = 0
    n = zeta.shape[0]
    k2d = np.sqrt(ki**2 + kj**2)*(n*dx)/(2.*np.pi)
    kc = 8
    dk = 3
    gen_response = np.exp(-0.5*(k2d-kc)**2/dk**2)
    if np.array(gen).size==1:
        f += gen*gen_response*zeta
    else:
        dims = u.shape
        ni, nj = (dims[0], dims[1])
        f += np.tile(gen, (ni, nj, 1))*gen_response*zeta
    ##dissipation term:
    f -= diss*(ki**2+kj**2)*zeta
    return f


###some feature-space diagnostics
def vortex_center(X):
    u, v = (X[:, :, 0], X[:, :, 1])
    ni, nj = u.shape
    zeta = (np.roll(v, -1, axis=0) - np.roll(v, 1, axis=0) - np.roll(u, -1, axis=1) + np.roll(u, 1, axis=1))/2.0
    zmax = -999
    ic, jc = (-1, -1)
    ##coarse search
    buff = 6
    for i in range(buff, ni-buff):
        for j in range(buff, nj-buff):
            z = np.sum(zeta[i-buff:i+buff, j-buff:j+buff])
            if z > zmax:
                zmax = z
                ic, jc = (i, j)
    return np.array([ic, jc])

def vortex_intensity(X):
    u, v = (X[:, :, 0], X[:, :, 1])
    wind = np.sqrt(u**2+v**2)
    return np.max(wind)

def vortex_size(X):
    center = vortex_center(X)
    wind = np.sqrt(X[:, :, 0]**2 + X[:, :, 1]**2)
    ni, nj = wind.shape
    nr = 30
    wind_min = 15  ##15 for no bg flow cases!
    wind_rad = np.zeros(nr)
    count_rad = np.zeros(nr)
    for i in range(-nr, nr+1):
        for j in range(-nr, nr+1):
            r = int(np.sqrt(i**2+j**2))
            if (r<nr):
                wind_rad[r] += wind[int(center[0]+i)%ni, int(center[1]+j)%nj]
                count_rad[r] += 1
    wind_rad = wind_rad/count_rad
    if np.max(wind_rad)<wind_min:
        size = -1
    else:
        i1 = np.where(wind_rad>=wind_min)[0][-1] ###last point with wind > 35knot
        if i1==nr-1:
            size = i1
        else:
            size = i1 + (wind_rad[i1] - wind_min) / (wind_rad[i1] - wind_rad[i1+1])
    return size


def divergence(X, dx):
    uk, vk = (grid2spec(X[:, :, 0]), grid2spec(X[:, :, 1]))
    ki, kj = get_scaled_wn(uk, dx)
    divk = 1j*(ki*uk + kj*vk)
    div = spec2grid(divk)
    return div


