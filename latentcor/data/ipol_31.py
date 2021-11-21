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

def NB_value(tau, zratio1_1, zratio1_2, zratio2_1):
    zratio1 = zratio2 = numpy.full((2, 1), numpy.nan)
    zratio1[ : , 0] = [zratio1_1, zratio1_2]; zratio2[0, 0] = zratio2_1
    tau = tau * internal.r_switch.bound_switch(self = internal.r_switch, comb = "31", zratio1 = zratio1, zratio2 = zratio2)
    output = internal.r_sol.batch(self = internal.r_sol, K = tau, comb = "31", zratio1 = zratio1, zratio2 = zratio2, tol = 1e-8)
    return output

tau_grid = stats.norm.cdf(numpy.linspace(-1.8, 1.8, 41), scale = .8) * 2 - 1
zratio1_1_grid = zratio1_2_grid = zratio2_1_grid = stats.norm.cdf(numpy.linspace(-1.8, 1.8, 41), scale = .8)
points_NB = (tau_grid, zratio1_1_grid, zratio1_2_grid, zratio2_1_grid)
points_NB_meshgrid = numpy.meshgrid(*points_NB, indexing='ij')
points_NB_tau_grid = points_NB_meshgrid[0].flatten()
points_NB_zratio1_1_grid = points_NB_meshgrid[1].flatten()
points_NB_zratio1_2_grid = points_NB_meshgrid[2].flatten()
points_NB_zratio2_1_grid = points_NB_meshgrid[3].flatten()

def NB_par(i):
    out = NB_value(tau = points_NB_tau_grid[i], zratio1_1 = points_NB_zratio1_1_grid[i], \
                                                zratio1_2 = points_NB_zratio1_2_grid[i], zratio2_1 = points_NB_zratio2_1_grid[i])
    return out
value_NB = Parallel(n_jobs=48)(delayed(NB_par)(i) for i in range(len(points_NB_tau_grid)))
value_NB = numpy.array(value_NB, dtype=numpy.float32).reshape(points_NB_meshgrid[0].shape)
print(value_NB)

ipol_31 = RegularGridInterpolator(points_NB, value_NB)
with lzma.open(os.path.join(os.getcwd(), "ipol_31.xz"), "wb", preset = 9) as f:
    pickle.dump(ipol_31, f)