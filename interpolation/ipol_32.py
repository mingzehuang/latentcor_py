import os
import sys
sys.path.append('/scratch/user/sharkmanhmz/python_project_package')
"""sys.path.append('/scratch/user/sharkmanhmz/latentcor_py/latentcor')"""
sys.path.insert(0, os.path.abspath('../latentcor'))
import numpy
import latentcor
import pickle
import lzma
from scipy import stats
from scipy.interpolate._rgi import RegularGridInterpolator
from joblib import Parallel, delayed

def NT_value(tau, zratio1_1, zratio1_2, zratio2_1):
    zratio1 = zratio2 = numpy.full((2, 1), numpy.nan)
    zratio1[ : , 0] = [zratio1_1 * zratio1_2, zratio1_2]; zratio2[0, 0] = zratio2_1
    tau = tau * latentcor.r_switch.bound_switch(self = latentcor.r_switch, comb = "32", zratio1 = zratio1, zratio2 = zratio2)
    output = latentcor.r_sol.batch(self = latentcor.r_sol, K = tau,  comb = "32", zratio1 = zratio1, zratio2 = zratio2, tol = 1e-8)
    return output

tau_grid = numpy.array([-1, *stats.norm.cdf(numpy.linspace(-1.2, 1.2, 25), scale = .5) * 2 - 1, 1], dtype = numpy.float32)
zratio1_1_grid = zratio1_2_grid = numpy.array([0, *stats.norm.cdf(numpy.linspace(-1.8, 1.8, 25), scale = .8), 1], dtype = numpy.float32)
zratio2_1_grid = numpy.array([0, *stats.norm.cdf(numpy.linspace(.1, 2.5, 25)) * 2 - 1, 1], dtype = numpy.float32)
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
value_NT = Parallel(n_jobs=68)(delayed(NT_par)(i) for i in range(len(points_NT_tau_grid)))
value_NT = numpy.array(value_NT, dtype=numpy.float32).reshape(points_NT_meshgrid[0].shape)
print(value_NT)

ipol_32 = RegularGridInterpolator(points_NT, value_NT)
with lzma.open(os.path.join(os.getcwd(), "ipol_32.xz"), "wb", preset = 9) as f:
    pickle.dump(ipol_32, f)
