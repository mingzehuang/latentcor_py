import numpy
import os
import sys
sys.path.append('/scratch/user/sharkmanhmz/latentcor_py/latentcor')
import internal
import pickle
import lzma
from scipy import stats
from scipy.interpolate import RegularGridInterpolator
from joblib import Parallel, delayed

def TT_value(tau, zratio1_1, zratio2_1):
    zratio1 = zratio2 = numpy.full((2, 1), numpy.nan)
    zratio1[0, 0] = zratio1_1; zratio2[0, 0] = zratio2_1
    tau = tau * internal.r_switch.bound_switch(self = internal.r_switch, comb = "22", zratio1 = zratio1, zratio2 = zratio2)
    output = internal.r_sol.batch(self = internal.r_sol, K = tau, comb = "22", zratio1 = zratio1, zratio2 = zratio2, tol = 1e-8)
    return output

tau_grid = stats.norm.cdf(numpy.linspace(-1.8, 1.8, 41), scale = .8) * 2 - 1
zratio1_1_grid = zratio2_1_grid = stats.norm.cdf(numpy.linspace(.1, 2.5, 41)) * 2 - 1
points_TT = (tau_grid, zratio1_1_grid, zratio2_1_grid)
points_TT_meshgrid = numpy.meshgrid(*points_TT, indexing='ij')
points_TT_tau_grid = points_TT_meshgrid[0].flatten()
points_TT_zratio1_1_grid = points_TT_meshgrid[1].flatten()
points_TT_zratio2_1_grid = points_TT_meshgrid[2].flatten()

def TT_par(i):
    out = TT_value(tau = points_TT_tau_grid[i], zratio1_1 = points_TT_zratio1_1_grid[i], zratio2_1 = points_TT_zratio2_1_grid[i])
    return out
value_TT = Parallel(n_jobs=48)(delayed(TT_par)(i) for i in range(len(points_TT_tau_grid)))
value_TT = numpy.array(value_TT, dtype=numpy.float32).reshape(points_TT_meshgrid[0].shape)
print(value_TT)

ipol_22 = RegularGridInterpolator(points_TT, value_TT)
with lzma.open(os.path.join(os.getcwd(), "ipol_22.xz"), "wb", preset = 9) as f:
    pickle.dump(ipol_22, f)

