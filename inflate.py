import numpy as np
from scipy.special import gamma

def adaptive_inflation(inf_mean, inf_sd, yb, varb, nens, yo, varo, corr):
  inf_update = inf_mean.copy()
  inf_sd_update = inf_sd.copy()
  d2 = (yo - yb)**2
  inf_sd_2 = inf_sd**2
  rate = change_GA_IG(inf_mean, inf_sd_2)
  inf_update = linear_bayes(d2, varb, varo, inf_mean, corr, nens, rate)
  return inf_update, inf_sd
  # shape_old = rate / inf_mean - 1.0
  # if (shape_old <= 2.0):
  #   return inf_update, inf_sd
  # else:
  #   dens1 = compute_new_density(d2, nens, varb, varo, shape_old, rate, corr, inf_update+inf_sd)
  #   dens2 = compute_new_density(d2, nens, varb, varo, shape_old, rate, corr, inf_update)
  #   if ( np.abs(dens1) < 1e-10 or np.abs(dens2) < 1e-10):
  #     return inf_update, inf_sd
  #   ratio = dens1 / dens2
  #   omega = np.log(inf_update)/inf_update + 1.0/inf_update - np.log(inf_update+inf_sd)/inf_update - 1.0/(inf_update+inf_sd)
  #   rate_new = np.log(ratio) / omega
  #   shape_new = rate_new / inf_update - 1.0
  #   if (shape_new <= 2.0):
  #     return inf_update, inf_sd
  #   else:
  #     inf_sd_update = np.sqrt(rate_new**2 / ((shape_new - 1.0)**2 * (shape_new-2.0)))
  #     if (inf_sd_update > 1.05*inf_sd or inf_sd_update < 0.6):
  #       return inf_update, inf_sd
  #     return inf_update, inf_sd_update

def linear_bayes(d2, varb, varo, inf_mean, corr, nens, beta):
  inf_update = inf_mean.copy()
  fac1 = (1.0 + corr * (np.sqrt(inf_mean) - 1.0))**2
  fac2 = -1.0/nens
  if (fac1 < np.abs(fac2)):
    fac2 = 0.0
  theta_bar_2 = (fac1+fac2)*varb + varo
  like_bar = np.exp(-0.5 * d2 / theta_bar_2) / np.sqrt(2.0*np.pi*theta_bar_2)
  if (like_bar <= 0.0):
    return inf_mean
  deriv_theta = 0.5*varb*corr*(1.0-corr+corr*np.sqrt(inf_mean)) / np.sqrt(theta_bar_2*inf_mean)
  like_prime = like_bar * deriv_theta * (d2/theta_bar_2 - 1.0) / np.sqrt(theta_bar_2)
  if (like_prime == 0.0 or np.abs(like_prime)<1e-10):
    return inf_mean
  like_ratio = like_bar / like_prime
  a = 1.0 - inf_mean / beta
  b = like_ratio - 2.0 * inf_mean
  c = inf_mean**2 - like_ratio * inf_mean
  plus_root, minus_root = solve_quadratic(a, b, c)
  if(np.abs(minus_root - inf_mean) < np.abs(plus_root - inf_mean)):
    inf_update = minus_root
  else:
    inf_update = plus_root
  if (inf_update<=0):
    inf_update = inf_mean
  return inf_update


def compute_new_density(d2, nens, varb, varo, alpha, beta, corr, inf):
  exp_prior = - beta / inf
  fac1 = (1.0 + corr * (np.sqrt(inf) - 1.0))**2
  fac2 = -1.0 / nens
  if(fac1 < np.abs(fac2)):
    fac2 = 0.0
  theta = np.sqrt((fac1+fac2)*varb + varo)
  exp_like = -0.5 * d2 / theta**2
  dens = beta**alpha / gamma(alpha) * inf**(-alpha-1.0) / (np.sqrt(2.0*np.pi)*theta) * np.exp(exp_like + exp_prior)
  return dens


def change_GA_IG(mode, var):
  var_p1 = var
  var_p2 = var_p1*var
  var_p3 = var_p2*var
  mode_p1 = mode
  mode_p2 = mode_p1*mode
  mode_p3 = mode_p2*mode
  mode_p4 = mode_p3*mode
  mode_p5 = mode_p4*mode
  mode_p6 = mode_p5*mode
  mode_p7 = mode_p6*mode
  mode_p8 = mode_p7*mode
  mode_p9 = mode_p8*mode
  AA = mode_p4 * np.sqrt((var_p2 + 47.0*var*mode_p2 + 3.0*mode_p4) / var_p3)
  BB = 75.0*var_p2*mode_p5
  CC = 21.0*var*mode_p7
  DD = var_p3*mode_p3
  EE = (CC + BB + DD + mode_p9 + 6.0*np.sqrt(3.0)*AA*var_p3) / var_p3
  beta = (7.0*var*mode + mode_p3)/(3.0*var) + EE**(1.0/3.0)/3.0 + mode_p2*(var_p2 + 14.0*var*mode_p2 + mode_p4) / (3.0*var_p2*EE**(1.0/3.0))
  return beta

def solve_quadratic(a, b, c):
  scaling = np.max(np.abs(np.array([a, b, c])))
  a1 = a / scaling
  b1 = b / scaling
  c1 = c / scaling
  disc = np.sqrt(b1**2 - 4.0 * a1 * c1)
  if (b1 > 0):
    r1 = (-b1 - disc) / (2.0 * a1)
  else:
    r1 = (-b1 + disc) / (2.0 * a1)
  r2 = (c1 / a1) / r1
  return r1, r2

