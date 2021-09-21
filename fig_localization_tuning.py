import numpy as np
import matplotlib.pyplot as plt
from config import *

nrealize = 1

network_type = 1
update_s = 0
local_cutoff_try = (8, 12, 16, 20, 24, 28, 32, 40, 48, 64)
local_dampen_try = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1)

###plot detailed contours for each case separately
loc_sprd = (3,)
ns = (2, 3, 4, 5)
fig, ax = plt.subplots(len(ns), len(loc_sprd), figsize=(4*len(loc_sprd), 2*len(ns)))
i0, j0 = np.meshgrid(local_cutoff_try, local_dampen_try)
ii = i0.T
jj = j0.T
for k in range(len(ns)):
    for l in range(len(loc_sprd)):
        rmse = np.zeros((len(local_cutoff_try), len(local_dampen_try), nrealize))
        for realize in range(nrealize):
            dirname = 'localization_tuning/{:04d}/type{}'.format(realize+1, network_type)
            scenario = '/Lsprd{}/ns{}_u{}'.format(loc_sprd[l], ns[k], update_s)
            rmse[:, :, realize] = np.load(outdir+dirname+scenario+'/rmse.npy')

        c = ax[k,l].contourf(ii, jj, np.median(rmse, axis=2), 20, cmap='gist_ncar')
        plt.colorbar(c, ax=ax[k,l])
        ax[k,l].set_xscale('log')

        mean_rmse = np.median(rmse, axis=2)
        # print(mean_rmse)
        best_i = ii[np.where(mean_rmse==np.min(mean_rmse))]
        best_j = jj[np.where(mean_rmse==np.min(mean_rmse))]
        ax[k,l].plot(best_i, best_j, 'wx')
plt.tight_layout()

###plot summary of minimum error contours and overlay on one plot
# ns = (2, 3, 4, 5, 6, 7)

plt.savefig('out.pdf')

