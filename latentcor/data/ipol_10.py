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

def BC_value(tau, zratio1_1):
    zratio1 = zratio2 = numpy.full((2, 1), numpy.nan)
    zratio1[0, 0] = zratio1_1
    tau = tau * internal.r_switch.bound_switch(self = internal.r_switch, comb = "10", zratio1 = zratio1, zratio2 = zratio2)
    output = internal.r_sol.batch(self = internal.r_sol, K = tau, comb = "10", zratio1 = zratio1, zratio2 = zratio2, tol = 1e-8)
    return output

tau_grid = numpy.array(stats.norm.cdf(numpy.linspace(-1.2, 1.2, 50), scale = .5) * 2 - 1, dtype = numpy.float32)
zratio1_1_grid = numpy.array(stats.norm.cdf(numpy.linspace(-1.2, 1.2, 50), scale = .5), dtype = numpy.float32)
points_BC = (tau_grid, zratio1_1_grid)
points_BC_meshgrid = numpy.meshgrid(*points_BC, indexing='ij')
points_BC_tau_grid = points_BC_meshgrid[0].flatten()
points_BC_zratio1_1_grid = points_BC_meshgrid[1].flatten()

def BC_par(i):
    out = BC_value(tau = points_BC_tau_grid[i], zratio1_1 = points_BC_zratio1_1_grid[i])
    return out
value_BC = Parallel(n_jobs=96)(delayed(BC_par)(i) for i in range(len(points_BC_tau_grid)))
value_BC = numpy.array(value_BC, dtype=numpy.float32).reshape(points_BC_meshgrid[0].shape)
print(value_BC)

ipol_10 = RegularGridInterpolator(points_BC, value_BC)

with lzma.open(os.path.join(os.getcwd(), "ipol_10.xz"), "wb", preset = 9) as f:
    pickle.dump(ipol_10, f)
