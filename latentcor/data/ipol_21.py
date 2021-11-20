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

def TB_value(tau, zratio1_1, zratio2_1):
    zratio1 = zratio2 = numpy.full((2, 1), numpy.nan)
    zratio1[0, 0] = zratio1_1; zratio2[0, 0] = zratio2_1
    tau = tau * internal.r_switch.bound_switch(self = internal.r_switch, comb = "21", zratio1 = zratio1, zratio2 = zratio2)
    output = internal.r_sol.batch(self = internal.r_sol, K = tau, comb = "21", zratio1 = zratio1, zratio2 = zratio2, tol = 1e-8)
    return output

tau_grid = stats.norm.cdf(numpy.linspace(-1.8, 1.8, 41), scale = .8) * 2 - 1
zratio1_1_grid = zratio2_1_grid = stats.norm.cdf(numpy.linspace(-1.2, 1.2, 41), scale = .5)
points_TB = (tau_grid, zratio1_1_grid, zratio2_1_grid)
points_TB_meshgrid = numpy.meshgrid(*points_TB, indexing='ij')
points_TB_tau_grid = points_TB_meshgrid[0].flatten()
points_TB_zratio1_1_grid = points_TB_meshgrid[1].flatten()
points_TB_zratio2_1_grid = points_TB_meshgrid[2].flatten()

def TB_par(i):
    out = TB_value(tau = points_TB_tau_grid[i], zratio1_1 = points_TB_zratio1_1_grid[i], zratio2_1 = points_TB_zratio2_1_grid[i])
    return out
value_TB = Parallel(n_jobs=8)(delayed(TB_par)(i) for i in range(len(points_TB_tau_grid)))
value_TB = numpy.array(value_TB, dtype=numpy.float32).reshape(points_TB_meshgrid[0].shape)
print(value_TB)

ipol_21 = RegularGridInterpolator(points_TB, value_TB)
with lzma.open(os.path.join(sys.path[0],"ipol_21.xz"), "wb", preset = 9) as f:
    pickle.dump(ipol_21, f)