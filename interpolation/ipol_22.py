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

def TT_value(tau, zratio1_1, zratio2_1):
    zratio1 = zratio2 = numpy.full((2, 1), numpy.nan)
    zratio1[0, 0] = zratio1_1; zratio2[0, 0] = zratio2_1
    tau = tau * latentcor.r_switch.bound_switch(self = latentcor.r_switch, comb = "22", zratio1 = zratio1, zratio2 = zratio2)
    output = latentcor.r_sol.batch(self = latentcor.r_sol, K = tau, comb = "22", zratio1 = zratio1, zratio2 = zratio2, tol = 1e-8)
    return output

tau_grid = numpy.array([-1, *stats.norm.cdf(numpy.linspace(-1.2, 1.2, 41), scale = .5) * 2 - 1, 1], dtype = numpy.float32)
zratio1_1_grid = zratio2_1_grid = numpy.array([0, *stats.norm.cdf(numpy.linspace(.1, 2.5, 41)) * 2 - 1, 1], dtype = numpy.float32)
points_TT = (tau_grid, zratio1_1_grid, zratio2_1_grid)
points_TT_meshgrid = numpy.meshgrid(*points_TT, indexing='ij')
points_TT_tau_grid = points_TT_meshgrid[0].flatten()
points_TT_zratio1_1_grid = points_TT_meshgrid[1].flatten()
points_TT_zratio2_1_grid = points_TT_meshgrid[2].flatten()

def TT_par(i):
    out = TT_value(tau = points_TT_tau_grid[i], zratio1_1 = points_TT_zratio1_1_grid[i], zratio2_1 = points_TT_zratio2_1_grid[i])
    return out
value_TT = Parallel(n_jobs=72)(delayed(TT_par)(i) for i in range(len(points_TT_tau_grid)))
value_TT = numpy.array(value_TT, dtype=numpy.float32).reshape(points_TT_meshgrid[0].shape)
print(value_TT)

ipol_22 = RegularGridInterpolator(points_TT, value_TT)
with lzma.open(os.path.join(os.getcwd(), "ipol_22.xz"), "wb", preset = 9) as f:
    pickle.dump(ipol_22, f)

