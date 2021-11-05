
import numpy
import sys
sys.path.append("C:/Users/mingz/Documents/latentcor_py/latentcor_py/latentcor")
import internal
import latentcor
import gen_data
import get_tps
from scipy.interpolate import interpn
import bz2
import pickle
import zlib
import lzma
import tarfile
from scipy import stats

def BC_value(tau, zratio1_1):
    zratio1 = zratio2 = numpy.full((2, 1), numpy.nan)
    zratio1[0, : ] = zratio1_1
    tau = tau * internal.r_switch.bound_switch(self = internal.r_switch, comb = "10", zratio1 = zratio1, zratio2 = zratio2)
    output = internal.r_sol.batch(self = internal.r_sol, K = tau, comb = "10", zratio1 = zratio1, zratio2 = zratio2, tol = 1e-8)
    return output

def BB_value(tau, zratio1_1, zratio2_1):
    zratio1 = [zratio1_1, numpy.nan]; zratio2 = [zratio2_1, numpy.nan]
    tau = tau * internal.r_switch.bound_switch(self = internal.r_switch, comb = "11", zratio1 = zratio1, zratio2 = zratio2)
    output = internal.r_sol.batch(self = internal.r_sol, K = tau, comb = "11", zratio1 = zratio1, zratio2 = zratio2, tol = 1e-8)
    return output

def TC_value(tau, zratio1_1):
    zratio1 = [zratio1_1, numpy.nan]; zratio2 = [numpy.nan, numpy.nan]
    tau = tau * internal.r_switch.bound_switch(self = internal.r_switch, comb = "20", zratio1 = zratio1, zratio2 = zratio2)
    output = internal.r_sol.batch(self = internal.r_sol, K = tau, comb = "20", zratio1 = zratio1, zratio2 = zratio2, tol = 1e-8)
    return output

def TB_value(tau, zratio1_1, zratio2_1):
    zratio1 = [zratio1_1, numpy.nan]; zratio2 = [zratio2_1, numpy.nan]
    tau = tau * internal.r_switch.bound_switch(self = internal.r_switch, comb = "21", zratio1 = zratio1, zratio2 = zratio2)
    output = internal.r_sol.batch(self = internal.r_sol, K = tau, comb = "21", zratio1 = zratio1, zratio2 = zratio2, tol = 1e-8)
    return output

def TT_value(tau, zratio1_1, zratio2_1):
    zratio1 = [zratio1_1, numpy.nan]; zratio2 = [zratio2_1, numpy.nan]
    tau = tau * internal.r_switch.bound_switch(self = internal.r_switch, comb = "22", zratio1 = zratio1, zratio2 = zratio2)
    output = internal.r_sol.batch(self = internal.r_sol, K = tau, comb = "22", zratio1 = zratio1, zratio2 = zratio2, tol = 1e-8)
    return output

def NC_value(tau, zratio1_1, zratio1_2):
    zratio1 = [zratio1_1, zratio1_2]; zratio2 = [numpy.nan, numpy.nan]
    tau = tau * internal.r_switch.bound_switch(self = internal.r_switch, comb = "30", zratio1 = zratio1, zratio2 = zratio2)
    output = internal.r_sol.batch(self = internal.r_sol, K = tau, zratio1 = zratio1, zratio2 = zratio2, tol = 1e-8)
    return output

def NB_value(tau, zratio1_1, zratio1_2, zratio2_1):
    zratio1 = [zratio1_1, zratio1_2]; zratio2 = [zratio2_1, numpy.nan]
    tau = tau * internal.r_switch.bound_switch(self = internal.r_switch, comb = "31", zratio1 = zratio1, zratio2 = zratio2)
    output = internal.r_sol.batch(self = internal.r_sol, K = tau, comb = "31", zratio1 = zratio1, zratio2 = zratio2, tol = 1e-8)
    return output

def NT_value(tau, zratio1_1, zratio1_2, zratio2_1):
    zratio1 = [zratio1_1, zratio1_2]; zratio2 = [zratio2_1, numpy.nan]
    tau = tau * internal.r_switch.bound_switch(self = internal.r_switch, comb = "32", zratio1 = zratio1, zratio2 = zratio2)
    output = internal.r_sol.batch(self = internal.r_sol, K = tau,  comb = "32", zratio1 = zratio1, zratio2 = zratio2, tol = 1e-8)
    return output

def NN_value(tau, zratio1_1, zratio1_2, zratio2_1, zratio2_2):
    zratio1 = [zratio1_1, zratio1_2]; zratio2 = [zratio2_1, zratio2_2]
    tau = tau * internal.r_switch.bound_switch(self = internal.r_switch, comb = "33", zratio1 = zratio1, zratio2 = zratio2)
    output = internal.r_sol.batch(self = internal.r_sol, K = tau, comb = "33", zratio1 = zratio1, zratio2 = zratio2, tol = 1e-8)
    return output

tau_grid = stats.norm.cdf(numpy.linspace(-1.2, 1.2, 41), scale = .5) * 2 - 1
zratio1_1_grid = stats.norm.cdf(numpy.linspace(-1.2, 1.2, 41), scale = .5)
points_BC = (tau_grid, zratio1_1_grid)
points_BC_meshgrid = numpy.meshgrid(*points_BC, indexing='ij')
points_BC_tau_grid = points_BC_meshgrid[0]
points_BC_zratio1_1_grid = points_BC_meshgrid[1]
value_BC = numpy.full(points_BC_tau_grid.shape, numpy.nan, dtype = numpy.int32)
for i in range(points_BC_tau_grid.shape[0]):
    for j in range(points_BC_tau_grid.shape[1]):
        """zratio1 = zratio2 = numpy.full((2, 1), numpy.nan)
        zratio1[0, : ] = points_BC_zratio1_1_grid[i, j]
        value_BC[i, j] = BC_value(tau = points_BC_tau_grid[i, j] / internal.r_switch.bound_switch(self = internal.r_switch, comb = "10", zratio1 = zratio1, zratio2 = zratio2), zratio1_1 = points_BC_zratio1_1_grid[i, j])
        """
        value_BC[i, j] = BC_value(tau = points_BC_tau_grid[i, j], zratio1_1 = points_BC_zratio1_1_grid[i, j]) * (10 ** 7)

print(value_BC[25,25])
"""Test"""
point = numpy.array([.3, .2])
print(interpn(points_BC, value_BC, point))

with lzma.open("value_BC.xz", "wb", preset = 9) as f:
    pickle.dump(value_BC, f)

with lzma.open("value_BC.xz", "rb") as f:
    value_BC = pickle.load(f)
    
print(value_BC)



"""
tau_grid = numpy.linspace(-.5, .5, by = .01)
zratio1_1_grid = zratio2_1_grid = numpy.linspace(.01, .99, by = .01)


tau_grid = numpy.linspace(-.99, .99, by = .01)
zratio1_1_grid = numpy.linspace(.01, .99, by = .01)


tau_grid = numpy.linspace(-.99, .99, by = .01)
zratio1_1_grid = zratio2_1_grid = numpy.linspace(.01, .99, by = .01)

    
tau_grid = numpy.linspace(-.99, .99, by = .01)
zratio1_1_grid = zratio2_1_grid = numpy.linspace(.01, .99, by = .01)


tau_grid = numpy.linspace(-.99, .99, by = .01)
zratio1_1_grid = zratio1_2_grid = numpy.linspace(.01, .99, by = .01)


tau_grid = numpy.linspace(-.99, .99, by = .01)
zratio1_1_grid = zratio1_2_grid = zratio2_1_grid = numpy.linspace(.01, .99, by = .01)

tau_grid = numpy.linspace(-.99, .99, by = .01)
zratio1_1_grid = zratio1_2_grid = zratio2_1_grid = numpy.linspace(.01, .99, by = .01)


tau_grid = numpy.linspace(-.99, .99, by = .01)
zratio1_1_grid = zratio1_2_grid = zratio2_1_grid = zratio2_2_grid = numpy.linspace(.01, .99, by = .01)

    
"""