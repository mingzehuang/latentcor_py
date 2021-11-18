
import numpy
import sys
sys.path.append("C:/Users/mingz/Documents/latentcor_py/latentcor_py/latentcor")
import internal
import latentcor
import gen_data
import get_tps
from scipy.interpolate import interpn
import pickle
import lzma
from scipy import stats
from scipy.interpolate import RegularGridInterpolator

def BC_value(tau, zratio1_1):
    zratio1 = zratio2 = numpy.full((2, 1), numpy.nan)
    zratio1[0, : ] = zratio1_1
    tau = tau * internal.r_switch.bound_switch(self = internal.r_switch, comb = "10", zratio1 = zratio1, zratio2 = zratio2)
    output = internal.r_sol.batch(self = internal.r_sol, K = tau, comb = "10", zratio1 = zratio1, zratio2 = zratio2, tol = 1e-8)
    return output

def BB_value(tau, zratio1_1, zratio2_1):
    zratio1 = zratio2 = numpy.full((2, 1), numpy.nan)
    zratio1[0, : ] = zratio1_1; zratio2[0, : ] = zratio2_1
    tau = tau * internal.r_switch.bound_switch(self = internal.r_switch, comb = "11", zratio1 = zratio1, zratio2 = zratio2)
    output = internal.r_sol.batch(self = internal.r_sol, K = tau, comb = "11", zratio1 = zratio1, zratio2 = zratio2, tol = 1e-8)
    return output

def TC_value(tau, zratio1_1):
    zratio1 = zratio2 = numpy.full((2, 1), numpy.nan)
    zratio1[0, : ] = zratio1_1
    tau = tau * internal.r_switch.bound_switch(self = internal.r_switch, comb = "20", zratio1 = zratio1, zratio2 = zratio2)
    output = internal.r_sol.batch(self = internal.r_sol, K = tau, comb = "20", zratio1 = zratio1, zratio2 = zratio2, tol = 1e-8)
    return output

def TB_value(tau, zratio1_1, zratio2_1):
    zratio1 = zratio2 = numpy.full((2, 1), numpy.nan)
    zratio1[0, : ] = zratio1_1; zratio2[0, : ] = zratio2_1
    tau = tau * internal.r_switch.bound_switch(self = internal.r_switch, comb = "21", zratio1 = zratio1, zratio2 = zratio2)
    output = internal.r_sol.batch(self = internal.r_sol, K = tau, comb = "21", zratio1 = zratio1, zratio2 = zratio2, tol = 1e-8)
    return output

def TT_value(tau, zratio1_1, zratio2_1):
    zratio1 = zratio2 = numpy.full((2, 1), numpy.nan)
    zratio1[0, : ] = zratio1_1; zratio2[0, : ] = zratio2_1
    tau = tau * internal.r_switch.bound_switch(self = internal.r_switch, comb = "22", zratio1 = zratio1, zratio2 = zratio2)
    output = internal.r_sol.batch(self = internal.r_sol, K = tau, comb = "22", zratio1 = zratio1, zratio2 = zratio2, tol = 1e-8)
    return output

def NC_value(tau, zratio1_1, zratio1_2):
    zratio1 = zratio2 = numpy.full((2, 1), numpy.nan)
    zratio1[ : , : ] = [zratio1_1, zratio1_2]
    tau = tau * internal.r_switch.bound_switch(self = internal.r_switch, comb = "30", zratio1 = zratio1, zratio2 = zratio2)
    output = internal.r_sol.batch(self = internal.r_sol, K = tau, zratio1 = zratio1, zratio2 = zratio2, tol = 1e-8)
    return output

def NB_value(tau, zratio1_1, zratio1_2, zratio2_1):
    zratio1 = zratio2 = numpy.full((2, 1), numpy.nan)
    zratio1[ : , : ] = [zratio1_1, zratio1_2]; zratio2[0, : ] = zratio2_1
    tau = tau * internal.r_switch.bound_switch(self = internal.r_switch, comb = "31", zratio1 = zratio1, zratio2 = zratio2)
    output = internal.r_sol.batch(self = internal.r_sol, K = tau, comb = "31", zratio1 = zratio1, zratio2 = zratio2, tol = 1e-8)
    return output

def NT_value(tau, zratio1_1, zratio1_2, zratio2_1):
    zratio1 = zratio2 = numpy.full((2, 1), numpy.nan)
    zratio1[ : , : ] = [zratio1_1, zratio1_2]; zratio2[0, : ] = zratio2_1
    tau = tau * internal.r_switch.bound_switch(self = internal.r_switch, comb = "32", zratio1 = zratio1, zratio2 = zratio2)
    output = internal.r_sol.batch(self = internal.r_sol, K = tau,  comb = "32", zratio1 = zratio1, zratio2 = zratio2, tol = 1e-8)
    return output

def NN_value(tau, zratio1_1, zratio1_2, zratio2_1, zratio2_2):
    zratio1 = zratio2 = numpy.full((2, 1), numpy.nan)
    zratio1[ : , : ] = [zratio1_1, zratio1_2]; zratio2[ : , : ] = [zratio2_1, zratio2_2]
    tau = tau * internal.r_switch.bound_switch(self = internal.r_switch, comb = "33", zratio1 = zratio1, zratio2 = zratio2)
    output = internal.r_sol.batch(self = internal.r_sol, K = tau, comb = "33", zratio1 = zratio1, zratio2 = zratio2, tol = 1e-8)
    return output

tau_grid = stats.norm.cdf(numpy.linspace(-1.2, 1.2, 41), scale = .5) * 2 - 1
zratio1_1_grid = stats.norm.cdf(numpy.linspace(-1.2, 1.2, 41), scale = .5)
points_BC = (tau_grid, zratio1_1_grid)
points_BC_meshgrid = numpy.meshgrid(*points_BC, indexing='ij')
points_BC_tau_grid = points_BC_meshgrid[0]
points_BC_zratio1_1_grid = points_BC_meshgrid[1]
value_BC = numpy.full(points_BC_tau_grid.shape, numpy.nan, dtype = numpy.float32)
for i in range(points_BC_tau_grid.shape[0]):
    for j in range(points_BC_tau_grid.shape[1]):
        value_BC[i, j] = BC_value(tau = points_BC_tau_grid[i, j], zratio1_1 = points_BC_zratio1_1_grid[i, j])
ipol_10 = RegularGridInterpolator(points_BC, value_BC)


"""Test"""
point = numpy.array([.3, .2])
print(ipol_10(point))

with lzma.open("ipol_10.xz", "wb", preset = 9) as f:
    pickle.dump(ipol_10, f)



tau_grid = stats.norm.cdf(numpy.linspace(-1.2, 1.2, 41), scale = .5) * 2 - 1
zratio1_1_grid = zratio2_1_grid = stats.norm.cdf(numpy.linspace(-1.2, 1.2, 41), scale = .5)
points_BB = (tau_grid, zratio1_1_grid, zratio2_1_grid)
points_BB_meshgrid = numpy.meshgrid(*points_BB, indexing='ij')
points_BB_tau_grid = points_BB_meshgrid[0]
points_BB_zratio1_1_grid = points_BB_meshgrid[1]
points_BB_zratio2_1_grid = points_BB_meshgrid[2]
value_BB = numpy.full(points_BB_tau_grid.shape, numpy.nan, dtype = numpy.float32)
for i in range(points_BB_tau_grid.shape[0]):
    for j in range(points_BB_tau_grid.shape[1]):
        for k in range(points_BB_tau_grid.shape[2]):
            value_BB[i, j, k] = BB_value(tau = points_BB_tau_grid[i, j, k], zratio1_1 = points_BB_zratio1_1_grid[i, j, k], zratio2_1 = points_BB_zratio2_1_grid[i, j, k])
ipol_11 = RegularGridInterpolator(points_BB, value_BB)

with lzma.open("ipol_11.xz", "wb", preset = 9) as f:
    pickle.dump(ipol_11, f)

tau_grid = stats.norm.cdf(numpy.linspace(-1.2, 1.2, 41), scale = .5) * 2 - 1
zratio1_1_grid = stats.norm.cdf(numpy.linspace(.1, 2.5, 41)) *2  - 1
points_TC = (tau_grid, zratio1_1_grid)
points_TC_meshgrid = numpy.meshgrid(*points_TC, indexing='ij')
points_TC_tau_grid = points_TC_meshgrid[0]
points_TC_zratio1_1_grid = points_TC_meshgrid[1]
value_TC = numpy.full(points_TC_tau_grid.shape, numpy.nan, dtype = numpy.float32)
for i in range(points_TC_tau_grid.shape[0]):
    for j in range(points_TC_tau_grid.shape[1]):
        value_TC[i, j] = TC_value(tau = points_TC_tau_grid[i, j], zratio1_1 = points_TC_zratio1_1_grid[i, j])
ipol_20 = RegularGridInterpolator(points_TC, value_TC)
with lzma.open("ipol_20.xz", "wb", preset = 9) as f:
    pickle.dump(ipol_20, f)


tau_grid = stats.norm.cdf(numpy.linspace(-1.8, 1.8, 41), scale = .8) * 2 - 1
zratio1_1_grid = zratio2_1_grid = stats.norm.cdf(numpy.linspace(-1.2, 1.2, 41), scale = .5)
points_TB = (tau_grid, zratio1_1_grid, zratio2_1_grid)
points_TB_meshgrid = numpy.meshgrid(*points_TB, indexing='ij')
points_TB_tau_grid = points_TB_meshgrid[0]
points_TB_zratio1_1_grid = points_TB_meshgrid[1]
points_TB_zratio2_1_grid = points_TB_meshgrid[2]
value_TB = numpy.full(points_TB_tau_grid.shape, numpy.nan, dtype = numpy.float32)
for i in range(points_TB_tau_grid.shape[0]):
    for j in range(points_TB_tau_grid.shape[1]):
        for k in range(points_TB_tau_grid.shape[2]):
            value_TB[i, j, k] = TB_value(tau = points_TB_tau_grid[i, j, k], zratio1_1 = points_TB_zratio1_1_grid[i, j, k], zratio2_1 = points_TB_zratio2_1_grid[i, j, k])
ipol_21 = RegularGridInterpolator(points_TB, value_TB)
with lzma.open("ipol_21.xz", "wb", preset = 9) as f:
    pickle.dump(ipol_21, f)

tau_grid = stats.norm.cdf(numpy.linspace(-1.8, 1.8, 41), scale = .8) * 2 - 1
zratio1_1_grid = zratio2_1_grid = stats.norm.cdf(numpy.linspace(.1, 2.5, 41)) * 2 - 1
points_TT = (tau_grid, zratio1_1_grid, zratio2_1_grid)
points_TT_meshgrid = numpy.meshgrid(*points_TT, indexing='ij')
points_TT_tau_grid = points_TT_meshgrid[0]
points_TT_zratio1_1_grid = points_TT_meshgrid[1]
points_TT_zratio2_1_grid = points_TT_meshgrid[2]
value_TT = numpy.full(points_TT_tau_grid.shape, numpy.nan, dtype = numpy.float32)
for i in range(points_TT_tau_grid.shape[0]):
    for j in range(points_TT_tau_grid.shape[1]):
        for k in range(points_TT_tau_grid.shape[2]):
            value_TT[i, j, k] = TT_value(tau = points_TT_tau_grid[i, j, k], zratio1_1 = points_TT_zratio1_1_grid[i, j, k], zratio2_1 = points_TT_zratio2_1_grid[i, j, k])
ipol_22 = RegularGridInterpolator(points_TT, value_TT)
with lzma.open("ipol_22.xz", "wb", preset = 9) as f:
    pickle.dump(ipol_22, f)

tau_grid = stats.norm.cdf(numpy.linspace(-1.2, 1.2, 41), scale = .5) * 2 - 1
zratio1_1_grid = zratio1_2_grid = stats.norm.cdf(numpy.linspace(-2.1, 2.1, 41))
points_NC = (tau_grid, zratio1_1_grid, zratio1_2_grid)
points_NC_meshgrid = numpy.meshgrid(*points_NC, indexing='ij')
points_NC_tau_grid = points_NC_meshgrid[0]
points_NC_zratio1_1_grid = points_NC_meshgrid[1]
points_TT_zratio2_1_grid = points_TT_meshgrid[2]
value_TT = numpy.full(points_TT_tau_grid.shape, numpy.nan, dtype = numpy.float32)
for i in range(points_TT_tau_grid.shape[0]):
    for j in range(points_TT_tau_grid.shape[1]):
        for k in range(points_TT_tau_grid.shape[2]):
            value_TT[i, j, k] = TT_value(tau = points_TT_tau_grid[i, j, k], zratio1_1 = points_TT_zratio1_1_grid[i, j, k], zratio2_1 = points_TT_zratio2_1_grid[i, j, k])

with lzma.open("value_TT.xz", "wb", preset = 9) as f:
    pickle.dump(value_TT, f)
with lzma.open("value_TT.xz", "rb") as f:
    value_TT = pickle.load(f)


"""
tau_grid = numpy.linspace(-.99, .99, by = .01)
zratio1_1_grid = zratio1_2_grid = zratio2_1_grid = numpy.linspace(.01, .99, by = .01)

tau_grid = numpy.linspace(-.99, .99, by = .01)
zratio1_1_grid = zratio1_2_grid = zratio2_1_grid = numpy.linspace(.01, .99, by = .01)


tau_grid = numpy.linspace(-.99, .99, by = .01)
zratio1_1_grid = zratio1_2_grid = zratio2_1_grid = zratio2_2_grid = numpy.linspace(.01, .99, by = .01)

    
"""