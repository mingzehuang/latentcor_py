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

def TC_value(tau, zratio1_1):
    zratio1 = zratio2 = numpy.full((2, 1), numpy.nan)
    zratio1[0, 0] = zratio1_1
    tau = tau * latentcor.r_switch.bound_switch(self = latentcor.r_switch, comb = "20", zratio1 = zratio1, zratio2 = zratio2)
    output = latentcor.r_sol.batch(self = latentcor.r_sol, K = tau, comb = "20", zratio1 = zratio1, zratio2 = zratio2, tol = 1e-8)
    return output

tau_grid = numpy.array([-1, *stats.norm.cdf(numpy.linspace(-1.2, 1.2, 41), scale = .5) * 2 - 1, 1], dtype = numpy.float32)
zratio1_1_grid = numpy.array([0, *stats.norm.cdf(numpy.linspace(.1, 2.5, 41)) * 2  - 1, 1], dtype = numpy.float32)
points_TC = (tau_grid, zratio1_1_grid)
points_TC_meshgrid = numpy.meshgrid(*points_TC, indexing='ij')
points_TC_tau_grid = points_TC_meshgrid[0].flatten()
points_TC_zratio1_1_grid = points_TC_meshgrid[1].flatten()

def TC_par(i):
    out = TC_value(tau = points_TC_tau_grid[i], zratio1_1 = points_TC_zratio1_1_grid[i])
    return out
value_TC = Parallel(n_jobs=72)(delayed(TC_par)(i) for i in range(len(points_TC_tau_grid)))
value_TC = numpy.array(value_TC, dtype=numpy.float32).reshape(points_TC_meshgrid[0].shape)
print(value_TC)

ipol_20 = RegularGridInterpolator(points_TC, value_TC)
with lzma.open(os.path.join(os.getcwd(), "ipol_20.xz"), "wb", preset = 9) as f:
    pickle.dump(ipol_20, f)