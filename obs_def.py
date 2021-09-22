import numpy as np

###observation network
def gen_network(network_type):
    if network_type==1:  ##global network
        nobs = 500
        obs_range = 100
    if network_type==2:  ##targeted network
        nobs = 6000
        obs_range = 30
    return nobs, obs_range

def gen_obs_loc(ni, nj, nv, nobs):
    Yloc2 = np.zeros((2, nobs))
    Yloc2[0, :] = np.random.uniform(0, ni, nobs)
    Yloc2[1, :] = np.random.uniform(0, nj, nobs)
    Yloc = np.zeros((3, nobs*nv))
    for k in range(nv):
        Yloc[0:2, k::nv] = Yloc2
        Yloc[2, k::nv] = k
    return Yloc

def obs_forward(X, Yloc):
    if Yloc[2, 0] == -1:  ##this is position obs
        from rankine_vortex import vortex_center
        Y = vortex_center(X)
    else:  ##this is normal wind obs
        Y = obs_interp2d(X, Yloc)
    return Y

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
    import matplotlib.pyplot as plt
    obsmin, obsmax = (-30, 30)
    cmap = [plt.cm.bwr(x) for x in np.linspace(0, 1, round(obsmax-obsmin)+1)]
    subset = np.where(np.logical_and(Yloc[2, :]==0, Ymask==1))
    color_ind = np.maximum(np.minimum(np.round(Y[subset]-obsmin), int(round(obsmax-obsmin))), 0).astype(int)
    ax.scatter(Yloc[0, subset], Yloc[1, subset], s=30, color=np.array(cmap)[color_ind, 0:3])
    ax.set_xlim([0, ni])
    ax.set_ylim([0, nj])
    return

