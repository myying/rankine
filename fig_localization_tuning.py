import numpy as np
import matplotlib.pyplot as plt
from config import *

local_cutoff_try = (8, 12, 16, 20, 24, 32, 40, 48, 64)
local_dampen_try = (0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1)
nrealize = 1

network_type = 1
loc_sprd = 3
ns = 1
update_s = 0

rmse = np.zeros((len(local_cutoff_try), len(local_dampen_try), nrealize))
for realize in range(nrealize):
    dirname = 'localization_tuning/{:04d}/type{}'.format(realize+1, network_type)
    scenario = '/Lsprd{}/ns{}_u{}'.format(loc_sprd, ns, update_s)
    rmse[:, :, realize] = np.load(outdir+dirname+scenario+'/rmse.npy')

plt.figure(figsize=(8,8))
ax = plt.subplot(111)

ii, jj = np.mgrid[0:len(local_cutoff_try), 0:len(local_dampen_try)]
c = ax.contourf(ii, jj, np.median(rmse, axis=2), cmap='jet')
plt.colorbar(c)
ax.set_xticks(np.arange(len(local_cutoff_try)))
ax.set_xticklabels(local_cutoff_try)
ax.set_yticks(np.arange(len(local_dampen_try)))
ax.set_yticklabels(local_dampen_try)

plt.savefig('out.pdf')

