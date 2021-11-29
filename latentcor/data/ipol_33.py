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

def NN_value(tau, zratio1_1, zratio1_2, zratio2_1, zratio2_2):
    zratio1 = zratio2 = numpy.full((2, 1), numpy.nan)
    zratio1[ : , 0] = [zratio1_1 * zratio1_2, zratio1_2]; zratio2[ : , 0] = [zratio2_1 * zratio2_2, zratio2_2]
    tau = tau * internal.r_switch.bound_switch(self = internal.r_switch, comb = "33", zratio1 = zratio1, zratio2 = zratio2)
    output = internal.r_sol.batch(self = internal.r_sol, K = tau, comb = "33", zratio1 = zratio1, zratio2 = zratio2, tol = 1e-8)
    return output

tau_grid = numpy.array(stats.norm.cdf(numpy.linspace(-1.8, 1.8, 13), scale = .8) * 2 - 1, dtype = numpy.float32)
zratio1_1_grid = zratio1_2_grid = zratio2_1_grid = zratio2_2_grid = numpy.array(stats.norm.cdf(numpy.linspace(-1.8, 1.8, 13), scale = .8), dtype = numpy.float32)
points_NN = (tau_grid, zratio1_1_grid, zratio1_2_grid, zratio2_1_grid, zratio2_2_grid)
points_NN_meshgrid = numpy.meshgrid(*points_NN, indexing='ij')
points_NN_tau_grid = points_NN_meshgrid[0].flatten()
points_NN_zratio1_1_grid = points_NN_meshgrid[1].flatten()
points_NN_zratio1_2_grid = points_NN_meshgrid[2].flatten()
points_NN_zratio2_1_grid = points_NN_meshgrid[3].flatten()
points_NN_zratio2_2_grid = points_NN_meshgrid[4].flatten()

def NN_par(i):
    out = NN_value(tau = points_NN_tau_grid[i], zratio1_1 = points_NN_zratio1_1_grid[i], \
                                                       zratio1_2 = points_NN_zratio1_2_grid[i], zratio2_1 = points_NN_zratio2_1_grid[i], zratio2_2 = points_NN_zratio2_2_grid[i])
    return out
value_NN = Parallel(n_jobs=80)(delayed(NN_par)(i) for i in range(len(points_NN_tau_grid)))
value_NN = numpy.array(value_NN, dtype=numpy.float32).reshape(points_NN_meshgrid[0].shape)
print(value_NN)

ipol_33 = RegularGridInterpolator(points_NN, value_NN)
with lzma.open(os.path.join(os.getcwd(), "ipol_33.xz"), "wb", preset = 9) as f:
    pickle.dump(ipol_33, f)
