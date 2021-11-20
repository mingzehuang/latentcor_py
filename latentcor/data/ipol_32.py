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

def NT_value(tau, zratio1_1, zratio1_2, zratio2_1):
    zratio1 = zratio2 = numpy.full((2, 1), numpy.nan)
    zratio1[ : , 0] = [zratio1_1, zratio1_2]; zratio2[0, 0] = zratio2_1
    tau = tau * internal.r_switch.bound_switch(self = internal.r_switch, comb = "32", zratio1 = zratio1, zratio2 = zratio2)
    output = internal.r_sol.batch(self = internal.r_sol, K = tau,  comb = "32", zratio1 = zratio1, zratio2 = zratio2, tol = 1e-8)
    return output

tau_grid = stats.norm.cdf(numpy.linspace(-1.8, 1.8, 2), scale = .8) * 2 - 1
zratio1_1_grid = zratio1_2_grid = stats.norm.cdf(numpy.linspace(-1.8, 1.8, 2), scale = .8)
zratio2_1_grid = stats.norm.cdf(numpy.linspace(.1, 2.5, 2))
points_NT = (tau_grid, zratio1_1_grid, zratio1_2_grid, zratio2_1_grid)
points_NT_meshgrid = numpy.meshgrid(*points_NT, indexing='ij')
points_NT_tau_grid = points_NT_meshgrid[0].flatten()
points_NT_zratio1_1_grid = points_NT_meshgrid[1].flatten()
points_NT_zratio1_2_grid = points_NT_meshgrid[2].flatten()
points_NT_zratio2_1_grid = points_NT_meshgrid[3].flatten()

def NT_par(i):
    out = NT_value(tau = points_NT_tau_grid[i], zratio1_1 = points_NT_zratio1_1_grid[i], \
                                                zratio1_2 = points_NT_zratio1_2_grid[i], zratio2_1 = points_NT_zratio2_1_grid[i])
    return out
value_NT = Parallel(n_jobs=8)(delayed(NT_par)(i) for i in range(len(points_NT_tau_grid)))
value_NT = numpy.array(value_NT, dtype=numpy.float32).reshape(points_NT_meshgrid[0].shape)
print(value_NT)

ipol_32 = RegularGridInterpolator(points_NT, value_NT)
with lzma.open(os.path.join(sys.path[0],"ipol_32.xz"), "wb", preset = 9) as f:
    pickle.dump(ipol_32, f)