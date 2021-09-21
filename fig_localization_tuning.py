import numpy as np
import matplotlib.pyplot as plt
from config import *

nrealize = 100

network_type = 1
update_s = 0
local_cutoff_try = (8, 12, 16, 20, 24, 28, 32, 40, 48, 64)
local_dampen_try = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1)
i0, j0 = np.meshgrid(local_cutoff_try, local_dampen_try)
ii = i0.T
jj = j0.T

###plot detailed contours for each case separately
loc_sprd = 3
ns = (2, 3, 4, 5, 6, 7)
colors = [plt.cm.gist_rainbow(x) for x in np.linspace(0, 1, len(ns))]

plt.figure(figsize=(5,4))
ax = plt.subplot(111)
for k in range(len(ns)):
    rmse = np.zeros((len(local_cutoff_try), len(local_dampen_try), nrealize))
    for realize in range(nrealize):
        dirname = 'localization_tuning/{:04d}/type{}'.format(realize+1, network_type)
        scenario = '/Lsprd{}/ns{}_u{}'.format(loc_sprd, ns[k], update_s)
        rmse[:, :, realize] = np.load(outdir+dirname+scenario+'/rmse.npy')
    mean_rmse = np.median(rmse, axis=2)
    best_i = ii[np.where(mean_rmse==np.min(mean_rmse))]
    best_j = jj[np.where(mean_rmse==np.min(mean_rmse))]
    rmse_contour = np.min(mean_rmse) + (np.max(mean_rmse) - np.min(mean_rmse))*0.01
    ax.contour(ii, jj, mean_rmse, (rmse_contour,), colors=[colors[k][0:3]], linewidths=3)
    ax.plot(best_i, best_j, color=colors[k][0:3], marker='*', markersize=8, label='Ns={}'.format(ns[k]))
ax.set_xticks(local_cutoff_try)
ax.set_xlim([5, 50])
ax.set_ylim([0.1, 1.05])
ax.grid()
ax.legend()

plt.savefig('out.pdf')

