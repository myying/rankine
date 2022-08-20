import numpy as np
outdir = 'output/'

##model state parameters
ni = 128    # number of grid points i, j directions
nj = 128
nv = 2     # number of variables, (u, v)
dx = 9000
dt = 3600
nt = 12

smalldt = 60
gen = 5e-5
diss = 3e3

### vortex parameters
Vbg = 5      ##background flow amplitude
Vmax = 35     ## maximum wind speed (vortex intensity)
Rmw = 5       ## radius of maximum wind (vortex size)

##ensemble parameters
nens = 20    ##ensemble size
loc_sprd = 10  ##position spread in prior ensemble
vmax_sprd = 0
size_sprd = 0
gen_ens = gen*np.ones(nens)

##obs network parameters
nobs = 5000    ##number of observations in entire domain
obs_range = 200  ##radius from vortex center where obs are available (will be assimilated)
obs_err_std = 3.0   ##measurement error
obs_err_power_law = 1
obs_t_intv = 3

