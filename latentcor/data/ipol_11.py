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

def BB_value(tau, zratio1_1, zratio2_1):
    zratio1 = zratio2 = numpy.full((2, 1), numpy.nan)
    zratio1[0, 0] = zratio1_1; zratio2[0, 0] = zratio2_1
    tau = tau * internal.r_switch.bound_switch(self = internal.r_switch, comb = "11", zratio1 = zratio1, zratio2 = zratio2)
    output = internal.r_sol.batch(self = internal.r_sol, K = tau, comb = "11", zratio1 = zratio1, zratio2 = zratio2, tol = 1e-8)
    return output

tau_grid = numpy.array(stats.norm.cdf(numpy.linspace(-1.2, 1.2, 50), scale = .5) * 2 - 1, dtype = numpy.float32)
zratio1_1_grid = zratio2_1_grid = numpy.array(stats.norm.cdf(numpy.linspace(-1.2, 1.2, 50), scale = .5), dtype = numpy.float32)
points_BB = (tau_grid, zratio1_1_grid, zratio2_1_grid)
points_BB_meshgrid = numpy.meshgrid(*points_BB, indexing='ij')
points_BB_tau_grid = points_BB_meshgrid[0].flatten()
points_BB_zratio1_1_grid = points_BB_meshgrid[1].flatten()
points_BB_zratio2_1_grid = points_BB_meshgrid[2].flatten()

def BB_par(i):
    out = BB_value(tau = points_BB_tau_grid[i], zratio1_1 = points_BB_zratio1_1_grid[i], zratio2_1 = points_BB_zratio2_1_grid[i])
    return out
value_BB = Parallel(n_jobs=96)(delayed(BB_par)(i) for i in range(len(points_BB_tau_grid)))
value_BB = numpy.array(value_BB, dtype=numpy.float32).reshape(points_BB_meshgrid[0].shape)
print(value_BB)

ipol_11 = RegularGridInterpolator(points_BB, value_BB)

with lzma.open(os.path.join(os.getcwd(), "ipol_11.xz"), "wb", preset = 9) as f:
    pickle.dump(ipol_11, f)
