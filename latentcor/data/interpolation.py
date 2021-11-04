import numpy
import internal
import latentcor
import gen_data
import get_tps
from numba import vectorize, float32

@vectorize([float32(float32, float32)], target = 'parallel')

def BC_value(tau, zratio1_1):
    zratio1 = [zratio1_1, numpy.nan]; zratio2 = [numpy.nan, numpy.nan]
    tau = tau * internal.r_switch.bound_switch(self = internal.r_switch, comb = "10", zratio1 = zratio1, zratio2 = zratio2)
    output = internal.r_sol.batch(self = internal.r_sol, K = tau, zratio1 = zratio1, zratio2 = zratio2, tol = 1e-8)
    return output

def BB_value(tau, zratio1_1, zratio2_1):
    zratio1 = [zratio1_1, numpy.nan]; zratio2 = [zratio2_1, numpy.nan]
    tau = tau * internal.r_switch.bound_switch(self = internal.r_switch, comb = "11", zratio1 = zratio1, zratio2 = zratio2)
    output = internal.r_sol.batch(self = internal.r_sol, K = tau, zratio1 = zratio1, zratio2 = zratio2, tol = 1e-8)
    return output

def TC_value(tau, zratio1_1):
    zratio1 = [zratio1_1, numpy.nan]; zratio2 = [numpy.nan, numpy.nan]
    tau = tau * internal.r_switch.bound_switch(self = internal.r_switch, comb = "20", zratio1 = zratio1, zratio2 = zratio2)
    output = internal.r_sol.batch(self = internal.r_sol, K = tau, zratio1 = zratio1, zratio2 = zratio2, tol = 1e-8)
    return output

def TB_value(tau, zratio1_1, zratio2_1):
    zratio1 = [zratio1_1, numpy.nan]; zratio2 = [zratio2_1, numpy.nan]
    tau = tau * internal.r_switch.bound_switch(self = internal.r_switch, comb = "21", zratio1 = zratio1, zratio2 = zratio2)
    output = internal.r_sol.batch(self = internal.r_sol, K = tau, zratio1 = zratio1, zratio2 = zratio2, tol = 1e-8)
    return output

def TT_value(tau, zratio1_1, zratio2_1):
    zratio1 = [zratio1_1, numpy.nan]; zratio2 = [zratio2_1, numpy.nan]
    tau = tau * internal.r_switch.bound_switch(self = internal.r_switch, comb = "22", zratio1 = zratio1, zratio2 = zratio2)
    output = internal.r_sol.batch(self = internal.r_sol, K = tau, zratio1 = zratio1, zratio2 = zratio2, tol = 1e-8)
    return output

def NC_value(tau, zratio1_1, zratio1_2):
    zratio1 = [zratio1_1, zratio1_2]; zratio2 = [numpy.nan, numpy.nan]
    tau = tau * internal.r_switch.bound_switch(self = internal.r_switch, comb = "30", zratio1 = zratio1, zratio2 = zratio2)
    output = internal.r_sol.batch(self = internal.r_sol, K = tau, zratio1 = zratio1, zratio2 = zratio2, tol = 1e-8)
    return output

def NB_value(tau, zratio1_1, zratio1_2, zratio2_1):
    zratio1 = [zratio1_1, zratio1_2]; zratio2 = [zratio2_1, numpy.nan]
    tau = tau * internal.r_switch.bound_switch(self = internal.r_switch, comb = "31", zratio1 = zratio1, zratio2 = zratio2)
    output = internal.r_sol.batch(self = internal.r_sol, K = tau, zratio1 = zratio1, zratio2 = zratio2, tol = 1e-8)
    return output

def NT_value(tau, zratio1_1, zratio1_2, zratio2_1):
    zratio1 = [zratio1_1, zratio1_2]; zratio2 = [zratio2_1, numpy.nan]
    tau = tau * internal.r_switch.bound_switch(self = internal.r_switch, comb = "32", zratio1 = zratio1, zratio2 = zratio2)
    output = internal.r_sol.batch(self = internal.r_sol, K = tau, zratio1 = zratio1, zratio2 = zratio2, tol = 1e-8)
    return output

def NN_value(tau, zratio1_1, zratio1_2, zratio2_1, zratio2_2):
    zratio1 = [zratio1_1, zratio1_2]; zratio2 = [zratio2_1, zratio2_2]
    tau = tau * internal.r_switch.bound_switch(self = internal.r_switch, comb = "33", zratio1 = zratio1, zratio2 = zratio2)
    output = internal.r_sol.batch(self = internal.r_sol, K = tau, zratio1 = zratio1, zratio2 = zratio2, tol = 1e-8)
    return output

tau_grid = numpy.linspace(-.5, .5, by = .01)
zratio1_1_grid = numpy.linspace(.01, .99, by = .01)
points_BC = (tau_grid, zratio1_1_grid)
values_BC = BC_value(*numpy.meshgrid(*points_BC, indexing='ij'))

tau_grid = numpy.linspace(-.5, .5, by = .01)
zratio1_1_grid = zratio2_1_grid = numpy.linspace(.01, .99, by = .01)
points_BB = (tau_grid, zratio1_1_grid, zratio2_1_grid)
values_BB = BB_value(*numpy.meshgrid(*points_BB, indexing='ij'))

tau_grid = numpy.linspace(-.99, .99, by = .01)
zratio1_1_grid = numpy.linspace(.01, .99, by = .01)
points_TC = (tau_grid, zratio1_1_grid)
values_TC = TC_value(*numpy.meshgrid(*points_TC, indexing='ij'))

tau_grid = numpy.linspace(-.99, .99, by = .01)
zratio1_1_grid = zratio2_1_grid = numpy.linspace(.01, .99, by = .01)
points_TB = (tau_grid, zratio1_1_grid, zratio2_1_grid)
values_TB = TB_value(*numpy.meshgrid(*points_TB, indexing='ij'))
    
tau_grid = numpy.linspace(-.99, .99, by = .01)
zratio1_1_grid = zratio2_1_grid = numpy.linspace(.01, .99, by = .01)
points_TT = (tau_grid, zratio1_1_grid, zratio2_1_grid)
values_TT = TT_value(*numpy.meshgrid(*points_TT, indexing='ij'))

tau_grid = numpy.linspace(-.99, .99, by = .01)
zratio1_1_grid = zratio1_2_grid = numpy.linspace(.01, .99, by = .01)
points_NC = (tau_grid, zratio1_1_grid, zratio1_2_grid)
values_NC = NC_value(*numpy.meshgrid(*points_NC, indexing='ij'))

tau_grid = numpy.linspace(-.99, .99, by = .01)
zratio1_1_grid = zratio1_2_grid = zratio2_1_grid = numpy.linspace(.01, .99, by = .01)
points_NB = (tau_grid, zratio1_1_grid, zratio1_2_grid, zratio2_1_grid)
values_NB = NB_value(*numpy.meshgrid(*points_NB, indexing='ij'))

tau_grid = numpy.linspace(-.99, .99, by = .01)
zratio1_1_grid = zratio1_2_grid = zratio2_1_grid = numpy.linspace(.01, .99, by = .01)
points_NT = (tau_grid, zratio1_1_grid, zratio1_2_grid, zratio2_1_grid)
values_NT = NT_value(*numpy.meshgrid(*points_NT, indexing='ij'))

tau_grid = numpy.linspace(-.99, .99, by = .01)
zratio1_1_grid = zratio1_2_grid = zratio2_1_grid = zratio2_2_grid = numpy.linspace(.01, .99, by = .01)
points_NN = (tau_grid, zratio1_1_grid, zratio1_2_grid, zratio2_1_grid, zratio2_2_grid)
values_NN = NN_value(*numpy.meshgrid(*points_NN, indexing='ij'))
    
