import numpy
import os
import sys
sys.path.append("C:/Users/mingz/Documents/latentcor_py/latentcor_py/latentcor")
import internal
import pickle
import lzma
from scipy import stats
from scipy.interpolate import RegularGridInterpolator
from joblib import Parallel, delayed

def NC_value(tau, zratio1_1, zratio1_2):
    zratio1 = zratio2 = numpy.full((2, 1), numpy.nan)
    zratio1[ : , 0] = [zratio1_1, zratio1_2]
    tau = tau * internal.r_switch.bound_switch(self = internal.r_switch, comb = "30", zratio1 = zratio1, zratio2 = zratio2)
    output = internal.r_sol.batch(self = internal.r_sol, K = tau, comb = "30", zratio1 = zratio1, zratio2 = zratio2, tol = 1e-8)
    return output

tau_grid = stats.norm.cdf(numpy.linspace(-1.2, 1.2, 41), scale = .5) * 2 - 1
zratio1_1_grid = zratio1_2_grid = stats.norm.cdf(numpy.linspace(-2.1, 2.1, 41))
points_NC = (tau_grid, zratio1_1_grid, zratio1_2_grid)
points_NC_meshgrid = numpy.meshgrid(*points_NC, indexing='ij')
points_NC_tau_grid = points_NC_meshgrid[0].flatten()
points_NC_zratio1_1_grid = points_NC_meshgrid[1].flatten()
points_NC_zratio1_2_grid = points_NC_meshgrid[2].flatten()

def NC_par(i):
    out = NC_value(tau = points_NC_tau_grid[i], zratio1_1 = points_NC_zratio1_1_grid[i], zratio1_2 = points_NC_zratio1_2_grid[i])
    return out
value_NC = Parallel(n_jobs=8)(delayed(NC_par)(i) for i in range(len(points_NC_tau_grid)))
value_NC = numpy.array(value_NC, dtype=numpy.float32).reshape(points_NC_meshgrid[0].shape)
print(value_NC)

ipol_30 = RegularGridInterpolator(points_NC, value_NC)
with lzma.open(os.path.join(sys.path[0],"ipol_30.xz"), "wb", preset = 9) as f:
    pickle.dump(ipol_30, f)