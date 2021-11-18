
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
    zratio1[ : , 0] = [zratio1_1, zratio1_2]
    tau = tau * internal.r_switch.bound_switch(self = internal.r_switch, comb = "30", zratio1 = zratio1, zratio2 = zratio2)
    output = internal.r_sol.batch(self = internal.r_sol, K = tau, comb = "30", zratio1 = zratio1, zratio2 = zratio2, tol = 1e-8)
    return output

def NB_value(tau, zratio1_1, zratio1_2, zratio2_1):
    zratio1 = zratio2 = numpy.full((2, 1), numpy.nan)
    zratio1[ : , 0] = [zratio1_1, zratio1_2]; zratio2[0, : ] = zratio2_1
    tau = tau * internal.r_switch.bound_switch(self = internal.r_switch, comb = "31", zratio1 = zratio1, zratio2 = zratio2)
    output = internal.r_sol.batch(self = internal.r_sol, K = tau, comb = "31", zratio1 = zratio1, zratio2 = zratio2, tol = 1e-8)
    return output

def NT_value(tau, zratio1_1, zratio1_2, zratio2_1):
    zratio1 = zratio2 = numpy.full((2, 1), numpy.nan)
    zratio1[ : , 0] = [zratio1_1, zratio1_2]; zratio2[0, : ] = zratio2_1
    tau = tau * internal.r_switch.bound_switch(self = internal.r_switch, comb = "32", zratio1 = zratio1, zratio2 = zratio2)
    output = internal.r_sol.batch(self = internal.r_sol, K = tau,  comb = "32", zratio1 = zratio1, zratio2 = zratio2, tol = 1e-8)
    return output

def NN_value(tau, zratio1_1, zratio1_2, zratio2_1, zratio2_2):
    zratio1 = zratio2 = numpy.full((2, 1), numpy.nan)
    zratio1[ : , 0] = [zratio1_1, zratio1_2]; zratio2[ : , 0] = [zratio2_1, zratio2_2]
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



tau_grid = stats.norm.cdf(numpy.linspace(-1.2, 1.2, 5), scale = .5) * 2 - 1
zratio1_1_grid = zratio2_1_grid = stats.norm.cdf(numpy.linspace(-1.2, 1.2, 5), scale = .5)
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


tau_grid = stats.norm.cdf(numpy.linspace(-1.8, 1.8, 5), scale = .8) * 2 - 1
zratio1_1_grid = zratio2_1_grid = stats.norm.cdf(numpy.linspace(-1.2, 1.2, 5), scale = .5)
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


tau_grid = stats.norm.cdf(numpy.linspace(-1.8, 1.8, 5), scale = .8) * 2 - 1
zratio1_1_grid = zratio2_1_grid = stats.norm.cdf(numpy.linspace(.1, 2.5, 5)) * 2 - 1
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


tau_grid = stats.norm.cdf(numpy.linspace(-1.2, 1.2, 5), scale = .5) * 2 - 1
zratio1_1_grid = zratio1_2_grid = stats.norm.cdf(numpy.linspace(-2.1, 2.1, 5))
points_NC = (tau_grid, zratio1_1_grid, zratio1_2_grid)
points_NC_meshgrid = numpy.meshgrid(*points_NC, indexing='ij')
points_NC_tau_grid = points_NC_meshgrid[0]
points_NC_zratio1_1_grid = points_NC_meshgrid[1]
points_NC_zratio1_2_grid = points_NC_meshgrid[2]
value_NC = numpy.full(points_NC_tau_grid.shape, numpy.nan, dtype = numpy.float32)
for i in range(points_NC_tau_grid.shape[0]):
    for j in range(points_NC_tau_grid.shape[1]):
        for k in range(points_NC_tau_grid.shape[2]):
            value_NC[i, j, k] = NC_value(tau = points_NC_tau_grid[i, j, k], zratio1_1 = points_NC_zratio1_1_grid[i, j, k], zratio1_2 = points_NC_zratio1_2_grid[i, j, k])
ipol_30 = RegularGridInterpolator(points_NC, value_NC)
with lzma.open("ipol_30.xz", "wb", preset = 9) as f:
    pickle.dump(ipol_30, f)


tau_grid = stats.norm.cdf(numpy.linspace(-1.8, 1.8, 2), scale = .8) * 2 - 1
zratio1_1_grid = zratio1_2_grid = zratio2_1_grid = stats.norm.cdf(numpy.linspace(-1.8, 1.8, 2), scale = .8)
points_NB = (tau_grid, zratio1_1_grid, zratio1_2_grid, zratio2_1_grid)
points_NB_meshgrid = numpy.meshgrid(*points_NB, indexing='ij')
points_NB_tau_grid = points_NB_meshgrid[0]
points_NB_zratio1_1_grid = points_NB_meshgrid[1]
points_NB_zratio1_2_grid = points_NB_meshgrid[2]
points_NB_zratio2_1_grid = points_NB_meshgrid[3]
value_NB = numpy.full(points_NB_tau_grid.shape, numpy.nan, dtype = numpy.float32)
for i in range(points_NB_tau_grid.shape[0]):
    for j in range(points_NB_tau_grid.shape[1]):
        for k in range(points_NB_tau_grid.shape[2]):
            for l in range(points_NB_tau_grid.shape[3]):
                value_NB[i, j, k, l] = NB_value(tau = points_NB_tau_grid[i, j, k, l], zratio1_1 = points_NB_zratio1_1_grid[i, j, k, l], \
                                                zratio1_2 = points_NB_zratio1_2_grid[i, j, k, l], zratio2_1 = points_NB_zratio2_1_grid[i, j, k, l])
ipol_31 = RegularGridInterpolator(points_NB, value_NB)
with lzma.open("ipol_31.xz", "wb", preset = 9) as f:
    pickle.dump(ipol_31, f)


tau_grid = stats.norm.cdf(numpy.linspace(-1.8, 1.8, 2), scale = .8) * 2 - 1
zratio1_1_grid = zratio1_2_grid = stats.norm.cdf(numpy.linspace(-1.8, 1.8, 2), scale = .8)
zratio2_1_grid = stats.norm.cdf(numpy.linspace(.1, 2.5, 2))
points_NT = (tau_grid, zratio1_1_grid, zratio1_2_grid, zratio2_1_grid)
points_NT_meshgrid = numpy.meshgrid(*points_NT, indexing='ij')
points_NT_tau_grid = points_NT_meshgrid[0]
points_NT_zratio1_1_grid = points_NT_meshgrid[1]
points_NT_zratio1_2_grid = points_NT_meshgrid[2]
points_NT_zratio2_1_grid = points_NT_meshgrid[3]
value_NT = numpy.full(points_NT_tau_grid.shape, numpy.nan, dtype = numpy.float32)
for i in range(points_NT_tau_grid.shape[0]):
    for j in range(points_NT_tau_grid.shape[1]):
        for k in range(points_NT_tau_grid.shape[2]):
            for l in range(points_NT_tau_grid.shape[3]):
                value_NT[i, j, k, l] = NT_value(tau = points_NT_tau_grid[i, j, k, l], zratio1_1 = points_NT_zratio1_1_grid[i, j, k, l], \
                                                zratio1_2 = points_NT_zratio1_2_grid[i, j, k, l], zratio2_1 = points_NT_zratio2_1_grid[i, j, k, l])
ipol_32 = RegularGridInterpolator(points_NT, value_NT)
with lzma.open("ipol_32.xz", "wb", preset = 9) as f:
    pickle.dump(ipol_32, f)


tau_grid = stats.norm.cdf(numpy.linspace(-1.8, 1.8, 2), scale = .8) * 2 - 1
zratio1_1_grid = zratio1_2_grid = zratio2_1_grid = zratio2_2_grid = stats.norm.cdf(numpy.linspace(-1.8, 1.8, 2), scale = .8)
points_NN = (tau_grid, zratio1_1_grid, zratio1_2_grid, zratio2_1_grid, zratio2_2_grid)
points_NN_meshgrid = numpy.meshgrid(*points_NN, indexing='ij')
points_NN_tau_grid = points_NN_meshgrid[0]
points_NN_zratio1_1_grid = points_NN_meshgrid[1]
points_NN_zratio1_2_grid = points_NN_meshgrid[2]
points_NN_zratio2_1_grid = points_NN_meshgrid[3]
points_NN_zratio2_2_grid = points_NN_meshgrid[4]
value_NN = numpy.full(points_NN_tau_grid.shape, numpy.nan, dtype = numpy.float32)
for i in range(points_NN_tau_grid.shape[0]):
    for j in range(points_NN_tau_grid.shape[1]):
        for k in range(points_NN_tau_grid.shape[2]):
            for l in range(points_NN_tau_grid.shape[3]):
                for m in range(points_NN_tau_grid.shape[4]):
                    value_NN[i, j, k, l, m] = NN_value(tau = points_NN_tau_grid[i, j, k, l, m], zratio1_1 = points_NN_zratio1_1_grid[i, j, k, l, m], \
                                                       zratio1_2 = points_NN_zratio1_2_grid[i, j, k, l, m], zratio2_1 = points_NN_zratio2_1_grid[i, j, k, l, m], zratio2_2 = points_NN_zratio2_2_grid[i, j, k, l, m])
ipol_33 = RegularGridInterpolator(points_NN, value_NN)
with lzma.open("ipol_33.xz", "wb", preset = 9) as f:
    pickle.dump(ipol_33, f)

