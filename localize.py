import numpy as np

def make_dist(ni, nj, nv, iX, jX, iobs, jobs, vobs):
  dist = np.zeros(ni*nj*nv)
  for v in range(nv):
    dist[v*ni*nj:(v+1)*ni*nj] = np.sqrt((iX - iobs)**2 + (jX - jobs)**2)
  return dist

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


def local_ms_factor(ns):
  # local_factor = np.ones(ns)
  if ns==1:
    local_factor = np.array([1.0])
  if ns==2:
    local_factor = np.array([1.5, 1.0])
  if ns==3:
    local_factor = np.array([1.8, 1.2, 1.0])
  if ns==4:
    local_factor = np.array([2.0, 1.6, 1.0, 0.8])
  if ns==5:
    local_factor = np.array([2.0, 1.6, 1.2, 1.0, 0.8])
  if ns==6:
    local_factor = np.array([2.0, 1.8, 1.2, 1.0, 0.8, 0.5])
  if ns==7:
    local_factor = np.array([2.0, 1.8, 1.5, 1.2, 1.0, 0.8, 0.5])
  if ns==8:
    local_factor = np.array([2.0, 1.8, 1.6, 1.4, 1.2, 1.0, 0.8, 0.5])
  return local_factor
