"""sys.path.append('/scratch/user/sharkmanhmz/python_project_package')"""
"""sys.path.append('/scratch/user/sharkmanhmz/latentcor_py/latentcor')"""
"""sys.path.insert(0, os.path.abspath('../latentcor'))"""

"""Put all related functions here to simplify path setting"""

import os
import sys
sys.path.insert(0, os.path.abspath('.'))
import numpy
from pandas import DataFrame
from statsmodels.stats.correlation_tools import corr_nearest
import seaborn
from matplotlib import pyplot
from scipy import stats
from scipy.optimize import fminbound
from joblib import Parallel, delayed
import lzma
import pickle
import pkg_resources
from sys import exit

class fromZtoX(object):
    """Switch between different copula"""
    def copula_switch(self, copula, z):
        method_name = copula
        method = getattr(self, method_name, lambda: 'Invalid copula')
        return method(self = fromZtoX, z = z)
    """Define copula for no copula case"""
    def no (self, z):
        return z
    """Define copula for exponential copula case"""
    def expo (self, z):
        return numpy.exp(z)
    """Define copula for cube copula case"""
    def cube (self, z):
        return z ** 3
    """Switch between different data tps""" 
    def tp_switch(self, tp, copula, z, xp):
        method_name = tp
        method = getattr(self, method_name, lambda: 'Invalid tp')
        return method(self = fromZtoX, u = fromZtoX.copula_switch(self = fromZtoX, copula = copula, z = z), xp = xp)
    """Define continuous data"""
    def con (self, u, xp):
        return u
    """Define binary data"""
    def bin (self, u, xp):
        q = numpy.quantile(u, xp[0]); x = numpy.zeros(len(u), dtype = numpy.int32); x[u > q] = 1
        return x
    """Define truncated data"""
    def tru (self, u, xp):
        q = numpy.quantile(u, xp[0]); x = numpy.zeros(len(u), dtype = numpy.double); x[u > q] = u[u > q] - q
        return x
    """Define ternary data"""
    def ter (self, u, xp):
        q = numpy.quantile(u, numpy.cumsum(xp)); x = numpy.ones(len(u), dtype = numpy.int32); x[u > numpy.repeat(q[1], len(u))] = 2; x[u <= numpy.repeat(q[0], len(u))] = 0
        return x


class Kendalltau(object):
    def n_x(self, x, n):
        x_info = numpy.unique(x, return_counts = True)
        if (len(x_info[0]) != n):
            x_counts = x_info[1]; t_x = x_counts[x_counts > 1]; out = numpy.sum(t_x * (t_x - 1) / 2)
        else:
            out = 0
        return out
    def kendalltau(i, X, X_tril_indices_row, X_tril_indices_col):
        x = X[ : , X_tril_indices_row[i]]; y = X[ : , X_tril_indices_col[i]]
        n = len(x); n0 = n * (n-1) / 2
        n_x = Kendalltau.n_x(self = Kendalltau, x = x, n = n)
        n_y = Kendalltau.n_x(self = Kendalltau, x = y, n = n)
        n_x_sqd = numpy.sqrt(n0 - n_x); n_y_sqd = numpy.sqrt(n0 - n_y)
        k_b = stats.kendalltau(x, y)[0]
        btoa = n_x_sqd * n_y_sqd /n0
        k_a = k_b * btoa
        return k_a
    def Kendalltau(self, X):
        X_tril_indices = numpy.tril_indices(X.shape[1], -1)
        X_tril_indices_row = X_tril_indices[0]
        X_tril_indices_col = X_tril_indices[1]
        tril_len = len(X_tril_indices_row)
        K_a_lower = Parallel(n_jobs=-1)(delayed(Kendalltau.kendalltau)(i, X, X_tril_indices_row, X_tril_indices_col) for i in range(tril_len))
        return numpy.array(K_a_lower)


class zratio(object):
    """Switch zratios calculation between different data tps"""
    def zratios_switch(self, x, tp):
        method_name = tp
        method = getattr(self, method_name, lambda: 'Invalid type')
        return method(self = zratio, x = x)
    """Define zratios for continous variable"""
    def con(self, x):
        out = numpy.full((2, x.shape[1]), numpy.nan)
        return out
    """Define zratios for binary variable"""
    def bin(self, x):
        out = numpy.full((2, x.shape[1]), numpy.nan)
        out[0, : ] = numpy.sum((x == 0), axis = 0) / x.shape[0]
        return out
    """Define zratios for truncated variable"""
    def tru(self, x):
        out = numpy.full((2, x.shape[1]), numpy.nan)
        out[0, : ] = numpy.sum((x == 0), axis = 0) / x.shape[0]
        return out
    """Define zratios for ternary variable"""
    def ter(self, x):
        out = numpy.row_stack((numpy.sum((x == 0), axis = 0) / x.shape[0], 1 - (numpy.sum((x == 2), axis = 0) / x.shape[0])))
        return out
    """Loop tps on all variables to calculate zratios"""
    def batch(self, X, tps):
        tps = numpy.array(tps, dtype = str, ndmin = 1)
        out = numpy.full((2, X.shape[1]), numpy.nan)
        for tp in numpy.unique(tps):
            out[ : , tps == tp] = zratio.zratios_switch(self = zratio, x = X[ : , tps == tp], tp = tp)
        return out

class r_sol(object):
    def bridge_switch(self, comb, r, zratio1, zratio2):
        method_name = comb
        method = getattr(self, "bridge_" + str(method_name), lambda: 'Invalid mixed types')
        return method(self = r_sol, r = r, zratio1 = zratio1, zratio2 = zratio2)
    def bridge_10(self, r, zratio1, zratio2):
        de1 = stats.norm.ppf(zratio1[0])
        mat1 = numpy.array([[1, r / numpy.sqrt(2)], [r / numpy.sqrt(2), 1]], dtype = numpy.double, ndmin = 2)
        res = numpy.double(4 * stats.multivariate_normal.cdf(x = [de1, 0], cov = mat1) - 2 * zratio1[0])
        return res
    def bridge_11(self, r, zratio1, zratio2):
        de1 = stats.norm.ppf(zratio1[0]); de2 = stats.norm.ppf(zratio2[0])
        mat1 = numpy.array([[1, r], [r, 1]], dtype = numpy.double, ndmin = 2)
        res = numpy.double(2 * (stats.multivariate_normal.cdf(x = [de1, de2], cov = mat1) - zratio1[0] * zratio2[0]))
        return res
    def bridge_20(self, r, zratio1, zratio2):
        de1 = stats.norm.ppf(zratio1[0])
        mat1 = numpy.array([[1, 1 / numpy.sqrt(2)], [1 / numpy.sqrt(2), 1]], dtype = numpy.double, ndmin = 2)
        mat2 = numpy.array([[1, 1 / numpy.sqrt(2), r / numpy.sqrt(2)], [1 / numpy.sqrt(2), 1, r], [r / numpy.sqrt(2), r, 1]], dtype = numpy.double, ndmin = 2)
        res = numpy.double(- 2 * stats.multivariate_normal.cdf(x = [- de1, 0], cov = mat1) + 4 * stats.multivariate_normal.cdf(x = [- de1, 0, 0], cov = mat2))
        return res
    def bridge_21(self, r, zratio1, zratio2):
        de1 = stats.norm.ppf(zratio1[0]); de2 = stats.norm.ppf(zratio2[0])
        mat1 = numpy.array([[1, - r, 1 / numpy.sqrt(2)], [- r, 1, - r / numpy.sqrt(2)], [1 / numpy.sqrt(2), - r / numpy.sqrt(2), 1]], dtype = numpy.double, ndmin = 2)
        mat2 = numpy.array([[1, 0, - 1 / numpy.sqrt(2)], [0, 1, - r / numpy.sqrt(2)], [- 1 / numpy.sqrt(2), - r / numpy.sqrt(2), 1]], dtype = numpy.double, ndmin = 2)
        res = numpy.double(2 * (1 - zratio1[0]) * zratio2[0] - 2 * stats.multivariate_normal.cdf(x = [- de1, de2, 0], cov = mat1) \
            - 2 * stats.multivariate_normal.cdf(x = [-de1, de2, 0], cov = mat2))
        return res
    def bridge_22(self, r, zratio1, zratio2):
        de1 = stats.norm.ppf(zratio1[0]); de2 = stats.norm.ppf(zratio2[0])
        mat1 = numpy.array([[1, 0 , 1 / numpy.sqrt(2), - r / numpy.sqrt(2)], [0, 1, - r / numpy.sqrt(2), 1 / numpy.sqrt(2)], \
               [1 / numpy.sqrt(2), - r / numpy.sqrt(2), 1, - r], [- r / numpy.sqrt(2), 1 / numpy.sqrt(2), - r, 1]], dtype = numpy.double, ndmin = 2)
        mat2 = numpy.array([[1, r, 1 / numpy.sqrt(2), r / numpy.sqrt(2)], [r, 1, r / numpy.sqrt(2), 1 / numpy.sqrt(2)], \
               [1 / numpy.sqrt(2), r /numpy.sqrt(2), 1, r], [r / numpy.sqrt(2), 1 / numpy.sqrt(2), r, 1]], dtype = numpy.double, ndmin = 2)
        res = numpy.double(- 2 * stats.multivariate_normal.cdf(x = [- de1, - de2, 0, 0], cov = mat1) \
            + 2 * stats.multivariate_normal.cdf(x = [- de1, - de2, 0, 0], cov = mat2))
        return res
    def bridge_30(self, r, zratio1, zratio2):
        de1 = stats.norm.ppf(zratio1)
        mat1 = numpy.array([[1, r / numpy.sqrt(2)], [r / numpy.sqrt(2), 1]], dtype = numpy.double, ndmin = 2)
        mat2 = numpy.array([[1, 0, r / numpy.sqrt(2)], [0, 1, - r / numpy.sqrt(2)], [r / numpy.sqrt(2), - r / numpy.sqrt(2), 1]], dtype = numpy.double, ndmin = 2)
        res = numpy.double(4 * stats.multivariate_normal.cdf(x = [de1[1], 0], cov = mat1) - 2 * zratio1[1] \
            + 4 * stats.multivariate_normal.cdf(x = [de1[0], de1[1], 0], cov = mat2) - 2 * zratio1[0] * zratio1[1])
        return res
    def bridge_31(self, r, zratio1, zratio2):
        de1 = stats.norm.ppf(zratio1); de2 = stats.norm.ppf(zratio2[0])
        mat1 = numpy.array([[1, r], [r, 1]], dtype = numpy.double, ndmin = 2)
        res = numpy.double(2 * stats.multivariate_normal.cdf(x = [de2, de1[1]], cov = mat1) * (1 - zratio1[0]) \
            - 2 * zratio1[1] * (zratio2[0] - stats.multivariate_normal.cdf(x = [de2, de1[0]], cov = mat1)))
        return res
    def bridge_32(self, r, zratio1, zratio2):
        de1 = stats.norm.ppf(zratio1); de2 = stats.norm.ppf(zratio2[0])
        mat1 = numpy.array([[1, 0, 0], [0, 1, r], [0, r, 1]], dtype = numpy.double, ndmin = 2)
        mat2 = numpy.array([[1, 0, 0, r / numpy.sqrt(2)], [0, 1, - r, r / numpy.sqrt(2)], [0, - r, 1, - 1 / numpy.sqrt(2)], [r / numpy.sqrt(2), r / numpy.sqrt(2), - 1 / numpy.sqrt(2), 1]], dtype = numpy.double, ndmin = 2)
        mat3 = numpy.array([[1, 0, r, r / numpy.sqrt(2)], [0, 1, 0, r / numpy.sqrt(2)], [r, 0, 1, 1 / numpy.sqrt(2)], [r / numpy.sqrt(2), r / numpy.sqrt(2), 1 / numpy.sqrt(2), 1]], dtype = numpy.double, ndmin = 2)
        res = numpy.double(- 2 * (1 - zratio1[0]) * zratio1[1] + 2 * stats.multivariate_normal.cdf(x = [- de1[0], de1[1], de2], cov = mat1) \
              + 2 * stats.multivariate_normal.cdf(x = [- de1[0], de1[1], - de2, 0], cov = mat2) + 2 * stats.multivariate_normal.cdf(x = [- de1[0], de1[1], - de2, 0], cov = mat3))
        return res
    def bridge_33(self, r, zratio1, zratio2):
        de1 = stats.norm.ppf(zratio1); de2 = stats.norm.ppf(zratio2)
        mat1 = numpy.array([[1, r], [r, 1]], dtype = numpy.double, ndmin = 2)
        res = numpy.double(2 * stats.multivariate_normal.cdf(x = [de1[1], de2[1]], cov = mat1) * stats.multivariate_normal.cdf(x = [- de1[0], - de2[0]], cov = mat1) \
            - 2 * (zratio1[1] - stats.multivariate_normal.cdf(x = [de1[1], de2[0]], cov = mat1)) * (zratio2[1] - stats.multivariate_normal.cdf(x = [de1[0], de2[1]], cov = mat1)))
        return res
    def solver(i, K, comb, zratio1, zratio2, tol):
            obj = lambda r: (r_sol.bridge_switch(self = r_sol, r = r, comb = comb, zratio1 = zratio1[ : , i], zratio2 = zratio2[ : , i]) - K[i]) ** 2
            res = fminbound(obj, -0.99, 0.99, xtol = tol)
            return res
    def batch(self, K, comb, zratio1, zratio2, tol):
        K_len = len(K)
        out = Parallel(n_jobs=-1)(delayed(r_sol.solver)(i, K, comb, zratio1, zratio2, tol) for i in range(K_len))
        return out

class r_switch(object):
    def bound_switch(self, comb, zratio1, zratio2):
        method_name = comb
        method = getattr(self, "bound_" + str(method_name), lambda: 'Invalid mixed types')
        return method(self = r_switch, zratio1 = zratio1, zratio2 = zratio2)
    def bound_10(self, zratio1, zratio2):
        return 2 * zratio1[0, : ] * (1 - zratio1[0, : ])
    def bound_11(self, zratio1, zratio2):
        return 2 * numpy.minimum.reduce([zratio1[0, : ], zratio2[0, : ]]) * (1 - numpy.maximum.reduce([zratio1[0, : ], zratio2[0, : ]]))
    def bound_20(self, zratio1, zratio2):
        return 1 - zratio1[0, : ] ** 2
    def bound_21(self, zratio1, zratio2):
        return 2 * numpy.maximum.reduce([zratio2[0, : ], 1 - zratio2[0, : ]]) * (1 - numpy.maximum.reduce([zratio2[0, : ], 1 - zratio2[0, : ], zratio1[0, : ]]))
    def bound_22(self, zratio1, zratio2):
        return 1 - numpy.maximum.reduce([zratio1[0, : ], zratio2[0, : ]]) ** 2
    def bound_30(self, zratio1, zratio2):
        return 2 * (zratio1[0, : ] * (1 - zratio1[0, : ]) + (1 - zratio1[1, : ]) * (zratio1[1, : ] - zratio1[0, : ]))
    def bound_31(self, zratio1, zratio2):
        return 2 * numpy.minimum.reduce([zratio1[0, : ] * (1 - zratio1[0, : ]) + (1 - zratio1[1, : ]) * (zratio1[1, : ] - zratio1[0, : ]), zratio2[0, : ] * (1 - zratio2[0, : ])])
    def bound_32(self, zratio1, zratio2):
        return 1 - numpy.maximum.reduce([zratio1[0, : ], zratio1[1, : ] - zratio1[0, : ], 1 - zratio1[1, : ], zratio2[0, : ]]) ** 2
    def bound_33(self, zratio1, zratio2):
        return 2 * numpy.minimum.reduce([zratio1[0, : ] * (1 - zratio1[0, : ]) + (1 - zratio1[1, : ]) * (zratio1[1, : ] - zratio1[0, : ]), \
                       zratio2[0, : ] * (1 - zratio2[0, : ]) + (1 - zratio2[1, : ]) * (zratio2[1, : ] - zratio2[0, : ])])
    def ipol_switch(self, comb, K, zratio1, zratio2):
        if comb == "10":
            out = ipol_10(numpy.column_stack((K, zratio1[0, : ])))
        elif comb == "11":
            out = ipol_11(numpy.column_stack((K, zratio1[0, : ], zratio2[0, : ])))
        elif comb == "20":
            out = ipol_20(numpy.column_stack((K, zratio1[0, : ])))
        elif comb == "21":
            out = ipol_21(numpy.column_stack((K, zratio1[0, : ], zratio2[0, : ])))
        elif comb == "22":
            out = ipol_22(numpy.column_stack((K, zratio1[0, : ], zratio2[0, : ])))           
        elif comb == "30":
            out = ipol_30(numpy.column_stack((K, zratio1[0, : ], zratio1[1, : ])))
        elif comb == "31":
            out = ipol_31(numpy.column_stack((K, zratio1[0, : ], zratio1[1, : ], zratio2[0, : ])))
        elif comb == "32":
            out = ipol_32(numpy.column_stack((K, zratio1[0, : ], zratio1[1, : ], zratio2[0, : ])))
        elif comb == "33":
            out = ipol_33(numpy.column_stack((K, zratio1[0, : ], zratio1[1, : ], zratio2[0, : ], zratio2[1, : ])))
        else:
            print("Unrecognized type.")
            exit()
        return out
    def r_ml(self, K, zratio1, zratio2, comb):
        if comb == "30" or comb == "31" or comb == "32" or comb == "33":
            zratio1[0, : ] = zratio1[0, : ] / zratio1[1, : ]
        if comb == "33":
            zratio2[0, : ] = zratio2[0, : ] / zratio2[1, : ]
        out = r_switch.ipol_switch(self = r_switch, comb = comb, K = K, zratio1 = zratio1, zratio2 = zratio2)
        return out
    def r_approx(self, K, zratio1, zratio2, comb, tol, ratio):
        bound = r_switch.bound_switch(self = r_switch, comb = comb, zratio1 = zratio1, zratio2 = zratio2).flatten()
        cutoff = numpy.abs(K) > ratio * bound
        if not any(cutoff):
            out = r_switch.r_ml(self = r_switch, K = K / bound, zratio1 = zratio1, zratio2 = zratio2, comb = comb)
        elif all(cutoff):
            out = r_sol.batch(self = r_sol, K = K, zratio1 = zratio1, zratio2 = zratio2, comb = comb, tol = tol)
        else:
            out = numpy.full(len(K), numpy.nan); revcutoff = numpy.logical_not(cutoff)
            out[cutoff] = r_sol.batch(self = r_sol, K = K[cutoff], zratio1 = zratio1[ : , cutoff], zratio2 = zratio2[ : , cutoff], comb = comb, tol = tol)
            out[revcutoff] = r_switch.r_ml(self = r_switch, K = K[revcutoff] / bound[revcutoff], zratio1 = zratio1[ : , revcutoff], zratio2 = zratio2[ : , revcutoff], comb = comb)
        return out


"""Here we start to generate interpolant"""
import numpy
import latentcor
import pickle
import lzma
from scipy import stats
from scipy.interpolate._rgi import RegularGridInterpolator
from joblib import Parallel, delayed

def NN_value(tau, zratio1_1, zratio1_2, zratio2_1, zratio2_2):
    zratio1 = zratio2 = numpy.full((2, 1), numpy.nan)
    zratio1[ : , 0] = [zratio1_1 * zratio1_2, zratio1_2]; zratio2[ : , 0] = [zratio2_1 * zratio2_2, zratio2_2]
    tau = tau * r_switch.bound_switch(self = r_switch, comb = "33", zratio1 = zratio1, zratio2 = zratio2)
    output = r_sol.batch(self = r_sol, K = tau, comb = "33", zratio1 = zratio1, zratio2 = zratio2, tol = 1e-8)
    return output

tau_grid = numpy.array([-1, *stats.norm.cdf(numpy.linspace(-1.8, 1.8, 13), scale = .8) * 2 - 1, 1], dtype = numpy.double)
zratio1_1_grid = zratio1_2_grid = zratio2_1_grid = zratio2_2_grid = numpy.array([0, *stats.norm.cdf(numpy.linspace(-1.8, 1.8, 13), scale = .8), 1], dtype = numpy.double)
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

"""Here we should set n_jobs=numbers of logical cores on your machine"""
value_NN = Parallel(n_jobs=68, backend = 'multiprocessing')(delayed(NN_par)(i) for i in range(len(points_NN_tau_grid)))
value_NN = numpy.array(value_NN, dtype=numpy.double).reshape(points_NN_meshgrid[0].shape)
print(value_NN)

ipol_33 = RegularGridInterpolator(points_NN, value_NN)

"""Here the output will be compressed to an xz file"""
with lzma.open(os.path.join(os.getcwd(), "ipol_33.xz"), "wb", preset = 9) as f:
    pickle.dump(ipol_33, f)
