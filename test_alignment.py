import numpy as np
from scipy import interpolate, optimize


def cost_function(ni, nj, nv, Xb, H, obs, obserr, iD, jD):
    nobs = obs.size
    R = np.eye(nobs) * obserr**2
    R_inv = np.linalg.inv(R)
    Xb1 = Xb.copy()
    for v in range(nv):
        Xb1[v*ni*nj:(v+1)*ni*nj] = deformation(ni, nj, Xb[v*ni*nj:(v+1)*ni*nj], iD, jD)
    innov = np.matmul(H, Xb1.T) - obs
    cost1 = 0.5*np.matmul(np.matmul(innov.T, R_inv), innov)
    return cost1


def displace_vector(ni, nj, nv, Xb, H, obs, obserr):
    nobs = obs.size
    R = np.eye(nobs) * obserr**2
    R_inv = np.linalg.inv(R)
    D = np.array([0, 0])

    def cost_function(D):
        iD = D[0]
        jD = D[1]
        Xb1 = Xb.copy()
        for v in range(nv):
            Xb1[v*ni*nj:(v+1)*ni*nj] = deformation(ni, nj, Xb[v*ni*nj:(v+1)*ni*nj], iD, jD)
        innov = np.matmul(H, Xb1.T) - obs
        cost1 = 0.5*np.matmul(np.matmul(innov.T, R_inv), innov)
        return cost1

    # def grad_function(D):
    #     iD = D[0]
    #     jD = D[1]
    #     Xb1 = Xb.copy()
    #     for v in range(nv):
    #         Xb1[v*ni*nj:(v+1)*ni*nj] = deformation(ni, nj, Xb[v*ni*nj:(v+1)*ni*nj], iD, jD)
    #     innov = np.matmul(H, Xb1.T) - obs
    #     gradJ = np.zeros(2)
    #     gradX = np.zeros((2, nv*ni*nj))
    #     for v in range(nv):
    #         gradi, gradj = grad(np.reshape(Xb1[v*ni*nj:(v+1)*ni*nj], (ni, nj)))
    #         gradX[0, v*ni*nj:(v+1)*ni*nj] = np.reshape(gradi, (ni*nj,))
    #         gradX[1, v*ni*nj:(v+1)*ni*nj] = np.reshape(gradj, (ni*nj,))
    #     gradJ = np.matmul(np.matmul(np.matmul(gradX, H.T), R_inv), innov)
    #     return gradJ

    # res = optimize.minimize(cost_function, D, jac='2-point', method='CG', options={'disp':True, 'eps':1})
    res = optimize.basinhopping(cost_function, D, minimizer_kwargs={'method':'BFGS'}, stepsize=5, disp=True, niter=30)
    iD = res.x[0]
    jD = res.x[1]

    return iD, jD


def gaussian_random_field(Pk, ni):
    def fftIndgen(n):
        a = range(0, n/2+1)
        b = range(1, n/2)
        b.reverse()
        b = [-i for i in b]
        return a + b
    def Pk2(kx, ky):
        if kx == 0 and ky == 0:
            return 0.0
        return np.sqrt(Pk(np.sqrt(kx**2 + ky**2)))
    noise = np.fft.fft2(np.random.normal(size=(ni, ni)))
    amplitude = np.zeros((ni, ni))
    for i, kx in enumerate(fftIndgen(ni)):
        for j, ky in enumerate(fftIndgen(ni)):
            amplitude[i, j] = Pk2(kx, ky)
    return np.fft.ifft2(noise * amplitude)


def random_vector(ni, nj, mean, spread, k_cutoff):
    Pk = lambda k: np.exp(-0.5 * k**2 / (k_cutoff/3.5)**2)
    ii = gaussian_random_field(Pk, ni) * spread + mean[0]
    jj = gaussian_random_field(Pk, ni) * spread + mean[1]
    iD = np.reshape(np.real(ii), (ni*nj,))
    jD = np.reshape(np.real(jj), (ni*nj,))
    return iD, jD


def grad(f):
    ni, nj = f.shape
    gradx = np.zeros((ni, nj))
    grady = np.zeros((ni, nj))
    gradx[1:ni-1, :] = 0.5*(f[2:ni, :] - f[0:ni-2, :])
    gradx[0, :] = f[1, :] - f[0, :]
    gradx[ni-1, :] = f[ni-1, :] - f[ni-2, :]
    grady[:, 1:ni-1] = 0.5*(f[:, 2:ni] - f[:, 0:ni-2])
    grady[:, 0] = f[:, 1] - f[:, 0]
    grady[:, ni-1] = f[:, ni-1] - f[:, ni-2]
    return gradx, grady


def div(u, v):
    ni, nj = u.shape
    div = np.zeros((ni, nj))
    div[1:ni-1, 1:nj-1] = 0.5*(u[2:ni, 1:nj-1] - u[0:ni-2, 1:nj-1]) + 0.5*(v[1:ni-1, 2:nj] - v[1:ni-1, 0:nj-2])
    div[0, 1:nj-1] = (u[1, 1:nj-1] - u[0, 1:nj-1]) + 0.5*(v[0, 2:nj] - v[0, 0:nj-2])
    div[ni-1, 1:nj-1] = (u[ni-1, 1:nj-1] - u[ni-2, 1:nj-1]) + 0.5*(v[ni-1, 2:nj] - v[ni-1, 0:nj-2])
    div[1:ni-1, 0] = 0.5*(u[2:ni, 0] - u[0:ni-2, 0]) + (v[1:ni-1, 1] - v[1:ni-1, 0])
    div[1:ni-1, nj-1] = 0.5*(u[2:ni, nj-1] - u[0:ni-2, nj-1]) + (v[1:ni-1, nj-1] - v[1:ni-1, nj-2])
    div[0, 0] = u[1, 0] - u[0, 0] + v[0, 1] - v[0, 0]
    div[ni-1, 0] = u[ni-1, 0] - u[ni-2, 0] + v[ni-1, 1] - v[ni-1, 0]
    div[0, nj-1] = u[1, nj-1] - u[0, nj-1] + v[0, nj-1] - v[0, nj-2]
    div[ni-1, nj-1] = u[ni-1, nj-1] - u[ni-2, nj-1] + v[ni-1, nj-1] - v[ni-1, nj-2]
    return div


def deformation(ni, nj, X, iD, jD):
    ii, jj = np.mgrid[0:ni, 0:nj]
    points = np.zeros((ni*nj, 2))
    points[:, 0] = np.reshape(ii[0:ni, 0:nj], (ni*nj,)) + iD
    points[:, 1] = np.reshape(jj[0:ni, 0:nj], (ni*nj,)) + jD
    xx = interpolate.griddata(points, X, (ii, jj), method='linear')
    xx1 = interpolate.griddata(points, X, (ii, jj), method='nearest')
    ind = np.where(np.isnan(xx))
    xx[ind] = xx1[ind]
    # xx = xx1
    Xd = np.reshape(xx[0:ni, 0:nj], (ni*nj,))

    return Xd


# def deformation(i, j, X, iD, jD):

