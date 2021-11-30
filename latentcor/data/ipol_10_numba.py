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
from numba import jit

def BC_value(tau, zratio1_1):
    zratio1 = zratio2 = numpy.full((2, 1), numpy.nan)
    zratio1[0, 0] = zratio1_1
    tau = tau * internal.r_switch.bound_switch(self = internal.r_switch, comb = "10", zratio1 = zratio1, zratio2 = zratio2)
    output = internal.r_sol.batch(self = internal.r_sol, K = tau, comb = "10", zratio1 = zratio1, zratio2 = zratio2, tol = 1e-8)
    return output

tau_grid = numpy.array([-1, *stats.norm.cdf(numpy.linspace(-1.2, 1.2, 41), scale = .5) * 2 - 1, 1], dtype = numpy.float32)
zratio1_1_grid = numpy.array([0, *stats.norm.cdf(numpy.linspace(-1.2, 1.2, 41), scale = .5), 1], dtype = numpy.float32)
points_BC = (tau_grid, zratio1_1_grid)
points_BC_meshgrid = numpy.meshgrid(*points_BC, indexing='ij')
points_BC_tau_grid = points_BC_meshgrid[0].flatten()
points_BC_zratio1_1_grid = points_BC_meshgrid[1].flatten()

def BC_par(i):
    out = BC_value(tau = points_BC_tau_grid[i], zratio1_1 = points_BC_zratio1_1_grid[i])
    return out

@jit(nonpython = True, parallel = True)
def fast_BC(BC_meshgrid): # Function is compiled and runs in machine code
    tau_grid = BC_meshgrid[0].flatten(); zratio1_1_grid = BC_meshgrid[1].flatten()
    for i in range(len(tau_grid)):
        value_BC = BC_value(tau = tau_grid[i], zratio1_1 = zratio1_1_grid[i])
    return value_BC.reshape(BC_meshgrid[0].shape)

fast_BC_value = fast_BC(points_BC_meshgrid)
print(fast_BC_value)

ipol_10_numba = RegularGridInterpolator(points_BC, fast_BC_value)

with lzma.open(os.path.join(os.getcwd(), "ipol_10_numba.xz"), "wb", preset = 9) as f:
    pickle.dump(ipol_10_numba, f)
