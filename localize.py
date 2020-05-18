import numpy as np


def GC(dist, cutoff):
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


def Gauss(dist, cutoff):
  loc = np.exp(-dist**2 / (cutoff / 3.5)**2 / 2)
  return loc
