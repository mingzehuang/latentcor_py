"""Internal functions"""

import numpy
from scipy import stats
from scipy.optimize import fminbound
import lzma
import pickle
from joblib import Parallel, delayed
from importlib import machinery
import os


"""ipol_10_file = pkg_resources.resource_stream('data', 'ipol_10.xz')"""
with lzma.open(os.path.join(os.getcwd(), "latentcor", "data", "ipol_10.xz"), "rb") as f:
    ipol_10 = pickle.load(f)

"""ipol_11_file = pkg_resources.resource_stream('data', 'ipol_11.xz')"""
with lzma.open(os.path.join(os.getcwd(), "latentcor", "data", "ipol_11.xz"), "rb") as f:
    ipol_11 = pickle.load(f)

"""ipol_20_file = pkg_resources.resource_stream('data', 'ipol_20.xz')"""
with lzma.open(os.path.join(os.getcwd(), "latentcor", "data", "ipol_20.xz"), "rb") as f:
    ipol_20 = pickle.load(f)

"""ipol_21_file = pkg_resources.resource_stream('data', 'ipol_21.xz')"""
with lzma.open(os.path.join(os.getcwd(), "latentcor", "data", "ipol_21.xz"), "rb") as f:
    ipol_21 = pickle.load(f)

"""ipol_22_file = pkg_resources.resource_stream('data', 'ipol_22.xz')"""
with lzma.open(os.path.join(os.getcwd(), "latentcor", "data", "ipol_22.xz"), "rb") as f:
    ipol_22 = pickle.load(f)

"""ipol_30_file = pkg_resources.resource_stream('data', 'ipol_30.xz')"""
with lzma.open(os.path.join(os.getcwd(), "latentcor", "data", "ipol_30.xz"), "rb") as f:
    ipol_30 = pickle.load(f)

"""ipol_31_file = pkg_resources.resource_stream('data', 'ipol_31.xz')"""
with lzma.open(os.path.join(os.getcwd(), "latentcor", "data", "ipol_31.xz"), "rb") as f:
    ipol_31 = pickle.load(f)

"""ipol_32_file = pkg_resources.resource_stream('data', 'ipol_32.xz')"""
with lzma.open(os.path.join(os.getcwd(), "latentcor", "data", "ipol_32.xz"), "rb") as f:
    ipol_32 = pickle.load(f)

"""ipol_33_file = pkg_resources.resource_stream('data', 'ipol_33.xz')"""
with lzma.open(os.path.join(os.getcwd(), "latentcor", "data", "ipol_33.xz"), "rb") as f:
    ipol_33 = pickle.load(f)

"""
with lzma.open(os.path.join(os.getcwd(), "latentcor", "data", "all_ipol.xz"), "rb") as f:
    ipol_10, ipol_11, ipol_20, ipol_21, ipol_22, ipol_30, ipol_31, ipol_32, ipol_33 = pickle.load(f)
"""

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
        q = numpy.quantile(u, xp[0]); x = numpy.zeros(len(u), dtype = numpy.float32); x[u > q] = u[u > q] - q
        return x
    """Define ternary data"""
    def ter (self, u, xp):
        q = numpy.quantile(u, numpy.cumsum(xp)); x = numpy.ones(len(u), dtype = numpy.int32); x[u > numpy.repeat(q[1], len(u))] = 2; x[u <= numpy.repeat(q[0], len(u))] = 0
        return x
"""Test class fromZtoX"""
"""print(fromZtoX.tp_switch(self = fromZtoX, tp = "tru", copula = "cube", z = numpy.random.standard_normal(100), xp = [0.3, 0.5]))

a = numpy.tril_indices(4, -1)
print(numpy.row_stack((a[0], a[1])).shape[1])"""

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

"""Test function n_x"""
"""print(Kendalltau.n_x(self = Kendalltau, x = [1, 3, 4, 5, 6, 7, 3, 2], n = 8))"""


"""Calculate zratios for X"""
class zratios(object):
    """Switch zratios calculation between different data tps"""
    def zratios_switch(self, x, tp):
        method_name = tp
        method = getattr(self, method_name, lambda: 'Invalid type')
        return method(self = zratios, x = x)
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
            out[ : , tps == tp] = zratios.zratios_switch(self = zratios, x = X[ : , tps == tp], tp = tp)
        return out
"""contry = numpy.array([fromZtoX.tp_switch(self = fromZtoX, tp = "con", copula = "no", z = numpy.random.standard_normal(100), xp = numpy.NaN)], dtype = float).T
print(zratios.con(self = zratios, x = contry))
bintry = numpy.array([fromZtoX.tp_switch(self = fromZtoX, tp = "bin", copula = "no", z = numpy.random.standard_normal(100), xp = [0.5])]).T
print(zratios.bin(self = zratios, x = bintry))
trutry = numpy.array([fromZtoX.tp_switch(self = fromZtoX, tp = "tru", copula = "no", z = numpy.random.standard_normal(100), xp = [0.5])]).T
print(zratios.tru(self = zratios, x = trutry))
tertry = numpy.array([fromZtoX.tp_switch(self = fromZtoX, tp = "ter", copula = "no", z = numpy.random.standard_normal(100), xp = [0.3, 0.5])]).T
print(zratios.ter(self = zratios, x = tertry))
print(zratios.ter(self = zratios, x = tertry))
contry2 = numpy.column_stack((contry, contry))
print(zratios.con(self = zratios, x = contry2))
bintry2 = numpy.column_stack((bintry, bintry))
print(zratios.bin(self = zratios, x = bintry2))
trutry2 = numpy.column_stack((trutry, trutry))
print(zratios.tru(self = zratios, x = trutry2))
tertry2 = numpy.column_stack((tertry, tertry))
print(zratios.ter(self = zratios, x = tertry2))
alldata = numpy.column_stack((contry, bintry, trutry, tertry))
print(alldata)
print(zratios.batch(self = zratios, X = alldata, tps = ["con", "bin", "tru", "ter"]))
alldata2 = numpy.column_stack((alldata, alldata))
print(zratios.batch(self = zratios, X = alldata2, tps = ["con", "bin", "tru", "ter", "con", "bin", "tru", "ter"]))
""""""test Kendall tau"""
"""print(Kendalltau.Kendalltau(self = Kendalltau, X = alldata2))
print(stats.kendalltau(contry, bintry)[0])"""

class r_sol(object):
    def bridge_switch(self, comb, r, zratio1, zratio2):
        method_name = comb
        method = getattr(self, "bridge_" + str(method_name), lambda: 'Invalid mixed types')
        return method(self = r_sol, r = r, zratio1 = zratio1, zratio2 = zratio2)
    def bridge_10(self, r, zratio1, zratio2):
        de1 = stats.norm.ppf(zratio1[0])
        mat1 = numpy.array([[1, r / numpy.sqrt(2)], [r / numpy.sqrt(2), 1]], dtype = numpy.float32, ndmin = 2)
        res = numpy.float32(4 * stats.multivariate_normal.cdf(x = [de1, 0], cov = mat1) - 2 * zratio1[0])
        return res
    def bridge_11(self, r, zratio1, zratio2):
        de1 = stats.norm.ppf(zratio1[0]); de2 = stats.norm.ppf(zratio2[0])
        mat1 = numpy.array([[1, r], [r, 1]], dtype = numpy.float32, ndmin = 2)
        res = numpy.float32(2 * (stats.multivariate_normal.cdf(x = [de1, de2], cov = mat1) - zratio1[0] * zratio2[0]))
        return res
    def bridge_20(self, r, zratio1, zratio2):
        de1 = stats.norm.ppf(zratio1[0])
        mat1 = numpy.array([[1, 1 / numpy.sqrt(2)], [1 / numpy.sqrt(2), 1]], dtype = numpy.float32, ndmin = 2)
        mat2 = numpy.array([[1, 1 / numpy.sqrt(2), r / numpy.sqrt(2)], [1 / numpy.sqrt(2), 1, r], [r / numpy.sqrt(2), r, 1]], dtype = numpy.float32, ndmin = 2)
        res = numpy.float32(- 2 * stats.multivariate_normal.cdf(x = [- de1, 0], cov = mat1) + 4 * stats.multivariate_normal.cdf(x = [- de1, 0, 0], cov = mat2))
        return res
    def bridge_21(self, r, zratio1, zratio2):
        de1 = stats.norm.ppf(zratio1[0]); de2 = stats.norm.ppf(zratio2[0])
        mat1 = numpy.array([[1, - r, 1 / numpy.sqrt(2)], [- r, 1, - r / numpy.sqrt(2)], [1 / numpy.sqrt(2), - r / numpy.sqrt(2), 1]], dtype = numpy.float32, ndmin = 2)
        mat2 = numpy.array([[1, 0, - 1 / numpy.sqrt(2)], [0, 1, - r / numpy.sqrt(2)], [- 1 / numpy.sqrt(2), - r / numpy.sqrt(2), 1]], dtype = numpy.float32, ndmin = 2)
        res = numpy.float32(2 * (1 - zratio1[0]) * zratio2[0] - 2 * stats.multivariate_normal.cdf(x = [- de1, de2, 0], cov = mat1) \
            - 2 * stats.multivariate_normal.cdf(x = [-de1, de2, 0], cov = mat2))
        return res
    def bridge_22(self, r, zratio1, zratio2):
        de1 = stats.norm.ppf(zratio1[0]); de2 = stats.norm.ppf(zratio2[0])
        mat1 = numpy.array([[1, 0 , 1 / numpy.sqrt(2), - r / numpy.sqrt(2)], [0, 1, - r / numpy.sqrt(2), 1 / numpy.sqrt(2)], \
               [1 / numpy.sqrt(2), - r / numpy.sqrt(2), 1, - r], [- r / numpy.sqrt(2), 1 / numpy.sqrt(2), - r, 1]], dtype = numpy.float32, ndmin = 2)
        mat2 = numpy.array([[1, r, 1 / numpy.sqrt(2), r / numpy.sqrt(2)], [r, 1, r / numpy.sqrt(2), 1 / numpy.sqrt(2)], \
               [1 / numpy.sqrt(2), r /numpy.sqrt(2), 1, r], [r / numpy.sqrt(2), 1 / numpy.sqrt(2), r, 1]], dtype = numpy.float32, ndmin = 2)
        res = numpy.float32(- 2 * stats.multivariate_normal.cdf(x = [- de1, - de2, 0, 0], cov = mat1) \
            + 2 * stats.multivariate_normal.cdf(x = [- de1, - de2, 0, 0], cov = mat2))
        return res
    def bridge_30(self, r, zratio1, zratio2):
        de1 = stats.norm.ppf(zratio1)
        mat1 = numpy.array([[1, r / numpy.sqrt(2)], [r / numpy.sqrt(2), 1]], dtype = numpy.float32, ndmin = 2)
        mat2 = numpy.array([[1, 0, r / numpy.sqrt(2)], [0, 1, - r / numpy.sqrt(2)], [r / numpy.sqrt(2), - r / numpy.sqrt(2), 1]], dtype = numpy.float32, ndmin = 2)
        res = numpy.float32(4 * stats.multivariate_normal.cdf(x = [de1[1], 0], cov = mat1) - 2 * zratio1[1] \
            + 4 * stats.multivariate_normal.cdf(x = [de1[0], de1[1], 0], cov = mat2) - 2 * zratio1[0] * zratio1[1])
        return res
    def bridge_31(self, r, zratio1, zratio2):
        de1 = stats.norm.ppf(zratio1); de2 = stats.norm.ppf(zratio2[0])
        mat1 = numpy.array([[1, r], [r, 1]], dtype = numpy.float32, ndmin = 2)
        res = numpy.float32(2 * stats.multivariate_normal.cdf(x = [de2, de1[1]], cov = mat1) * (1 - zratio1[0]) \
            - 2 * zratio1[1] * (zratio2[0] - stats.multivariate_normal.cdf(x = [de2, de1[0]], cov = mat1)))
        return res
    def bridge_32(self, r, zratio1, zratio2):
        de1 = stats.norm.ppf(zratio1); de2 = stats.norm.ppf(zratio2[0])
        mat1 = numpy.array([[1, 0, 0], [0, 1, r], [0, r, 1]], dtype = numpy.float32, ndmin = 2)
        mat2 = numpy.array([[1, 0, 0, r / numpy.sqrt(2)], [0, 1, - r, r / numpy.sqrt(2)], [0, - r, 1, - 1 / numpy.sqrt(2)], [r / numpy.sqrt(2), r / numpy.sqrt(2), - 1 / numpy.sqrt(2), 1]], dtype = numpy.float32, ndmin = 2)
        mat3 = numpy.array([[1, 0, r, r / numpy.sqrt(2)], [0, 1, 0, r / numpy.sqrt(2)], [r, 0, 1, 1 / numpy.sqrt(2)], [r / numpy.sqrt(2), r / numpy.sqrt(2), 1 / numpy.sqrt(2), 1]], dtype = numpy.float32, ndmin = 2)
        res = numpy.float32(- 2 * (1 - zratio1[0]) * zratio1[1] + 2 * stats.multivariate_normal.cdf(x = [- de1[0], de1[1], de2], cov = mat1) \
              + 2 * stats.multivariate_normal.cdf(x = [- de1[0], de1[1], - de2, 0], cov = mat2) + 2 * stats.multivariate_normal.cdf(x = [- de1[0], de1[1], - de2, 0], cov = mat3))
        return res
    def bridge_33(self, r, zratio1, zratio2):
        de1 = stats.norm.ppf(zratio1); de2 = stats.norm.ppf(zratio2)
        mat1 = numpy.array([[1, r], [r, 1]], dtype = numpy.float32, ndmin = 2)
        res = numpy.float32(2 * stats.multivariate_normal.cdf(x = [de1[1], de2[1]], cov = mat1) * stats.multivariate_normal.cdf(x = [- de1[0], - de2[0]], cov = mat1) \
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
"""Test bridge binary/continuous"""
"""print(r_sol.bridge_10(self = r_sol, r = .5, zratio1 = [.5, numpy.NaN], zratio2 = numpy.NaN))
print(r_sol.bridge_switch(self = r_sol, r = .5, comb = "10", zratio1 = [.5, numpy.NaN], zratio2 = numpy.NaN))
print(r_sol.bridge_11(self = r_sol, r = .5, zratio1 = [.5, numpy.NaN], zratio2 = [.5, numpy.NaN]))
print(r_sol.bridge_switch(self = r_sol, r = .5, comb = "11", zratio1 = [.5, numpy.NaN], zratio2 = [.5, numpy.NaN]))
print(r_sol.bridge_20(self = r_sol, r = .5, zratio1 = [.5, numpy.NaN], zratio2 = numpy.NaN))
print(r_sol.bridge_switch(self = r_sol, r = .5, comb = "20", zratio1 = [.5, numpy.NaN], zratio2 = numpy.NaN))
print(r_sol.bridge_21(self = r_sol, r = .5, zratio1 = [.5, numpy.NaN], zratio2  = [.5, numpy.NaN]))
print(r_sol.bridge_switch(self = r_sol, r = .5, comb = "21", zratio1 = [.5, numpy.NaN], zratio2 = [.5, numpy.NaN]))
print(r_sol.bridge_22(self = r_sol, r = .5, zratio1 = [.5, numpy.NaN], zratio2 = [.5, numpy.NaN]))
print(r_sol.bridge_switch(self = r_sol, r = .5, comb = "22", zratio1 = [.5, numpy.NaN], zratio2 = [.5, numpy.NaN]))
print(r_sol.bridge_30(self = r_sol, r = .5, zratio1 = [.3, .8], zratio2 = numpy.NaN))
print(r_sol.bridge_switch(self = r_sol, r = .5, comb = "30", zratio1 = [.3, .8], zratio2 = numpy.NaN))
print(r_sol.bridge_31(self = r_sol, r = .5, zratio1 = [.3, .8], zratio2 = [.5, numpy.NaN]))
print(r_sol.bridge_switch(self = r_sol, r = .5, comb = "31", zratio1 = [.3, .8], zratio2 = [.5, numpy.NaN]))
print(r_sol.bridge_32(self = r_sol, r = .5, zratio1 = [.3, .8], zratio2 = [.5, numpy.NaN]))
print(r_sol.bridge_switch(self = r_sol, r = .5, comb = "32", zratio1 = [.3, .8], zratio2 = [.5, numpy.NaN]))
print(r_sol.bridge_33(self = r_sol, r = .5, zratio1 = [.3, .8], zratio2 = [.3, .8]))
print(r_sol.bridge_switch(self = r_sol, r = .5, comb = "33", zratio1 = [.3, .8], zratio2 = [.3, .8]))"""
"""print(r_sol.obj(self = r_sol, r = .5, k = .3, comb = "10", zratio1 = .5, zratio2 = numpy.NaN))
print(r_sol.obj(self = r_sol, r = .5, k = .3, comb = "11", zratio1 = .5, zratio2 = .5))
print(r_sol.obj(self = r_sol, r = .5, k = .3, comb = "20", zratio1 = .5, zratio2 = numpy.NaN))
print(r_sol.obj(self = r_sol, r = .5, k = .3, comb = "21", zratio1 = .5, zratio2 = .5))
print(r_sol.obj(self = r_sol, r = .5, k = .3, comb = "22", zratio1 = .5, zratio2 = .5))
print(r_sol.obj(self = r_sol, r = .5, k = .3, comb = "30", zratio1 = [.3, .8], zratio2 = numpy.NaN))
print(r_sol.obj(self = r_sol, r = .5, k = .3, comb = "31", zratio1 = [.3, .8], zratio2 = .5))
print(r_sol.obj(self = r_sol, r = .5, k = .3, comb = "32", zratio1 = [.3, .8], zratio2 = .5))
print(r_sol.obj(self = r_sol, r = .5, k = .3, comb = "33", zratio1 = [.3, .8], zratio2 = [.3, .8]))"""

"""K = Kendalltau.Kendalltau(self = Kendalltau, X = bintry2)
zratios = zratios.batch(self = zratios, X = bintry2, tps = ["bin", "bin"])
X_tril_indices = numpy.tril_indices(bintry2.shape[1], -1)
X_tril_indices_row = X_tril_indices[0]; X_tril_indices_col = X_tril_indices[1]
zratio1 = zratios[ : , X_tril_indices_row]; zratio2 = zratios[ : , X_tril_indices_col]
print(K)
print(zratio1)
print(zratio1)
print(*zratio1[1])
print(zratio2)


print(stats.norm.ppf(zratio1))
obj = lambda r: (r_sol.bridge_switch(self = r_sol, r = r, comb = "11", zratio1 = zratio1[ : , 0], zratio2 = zratio2[ : , 0]) - K[0]) ** 2
print(obj)
print(obj(.5))
print((r_sol.bridge_switch(self = r_sol, r = .5, comb = "11", zratio1 = zratio1[ : , 0], zratio2 = zratio2[ : , 0]) - K) ** 2)
""""""
print(fminbound(obj, -.99, .99, xtol = .001))
print(r_sol.bridge_11(self = r_sol, r = .5, zratio1 = [.5, numpy.NaN], zratio2 = [.5, numpy.NaN]))
print(r_sol.bridge_11(self = r_sol, r = .5, zratio1 = zratio1, zratio2 = zratio2))
"""
"""print(r_sol.batch(self = r_sol, K = K, comb = "11", zratio1 = zratio1, zratio2 = zratio2, tol = 0.0001))
"""
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
"""print(r_switch.bound_10(self = r_switch, zratio1 = zratio1, zratio2 = numpy.NaN))
print(r_switch.bound_switch(self = r_switch, comb = "10", zratio1 = zratio1, zratio2 = numpy.NaN))
print(r_switch.bound_11(self = r_switch, zratio1 = zratio1, zratio2 = zratio2))
print(r_switch.bound_switch(self = r_switch, comb = "11", zratio1 = zratio1, zratio2 = zratio2))
print(r_switch.bound_20(self = r_switch, zratio1 = zratio1, zratio2 = numpy.NaN))
print(r_switch.bound_switch(self = r_switch, comb = "20", zratio1 = zratio1, zratio2 = numpy.NaN))
print(r_switch.bound_21(self = r_switch, zratio1 = zratio1, zratio2 = zratio2))
print(r_switch.bound_switch(self = r_switch, comb = "21", zratio1 = zratio1, zratio2 = zratio2))
print(r_switch.bound_22(self = r_switch, zratio1 = zratio1, zratio2 = zratio2))
print(r_switch.bound_switch(self = r_switch, comb = "22", zratio1 = zratio1, zratio2 = zratio2))
print(r_switch.bound_30(self = r_switch, zratio1 = numpy.array([[.3], [.8]]), zratio2 = numpy.NaN))
print(r_switch.bound_switch(self = r_switch, comb = "30", zratio1 = numpy.array([[.3], [.8]]), zratio2 = numpy.NaN))
print(r_switch.bound_31(self = r_switch, zratio1 = numpy.array([[.3], [.8]]), zratio2 = zratio2))
print(r_switch.bound_switch(self = r_switch, comb = "31", zratio1 = numpy.array([[.3], [.8]]), zratio2 = zratio2))
print(r_switch.bound_32(self = r_switch, zratio1 = numpy.array([[.3], [.8]]), zratio2 = zratio2))
print(r_switch.bound_switch(self = r_switch, comb = "32", zratio1 = numpy.array([[.3], [.8]]), zratio2 = zratio2))
print(r_switch.bound_33(self = r_switch, zratio1 = numpy.array([[.3], [.8]]), zratio2 = numpy.array([[.3], [.8]])))
print(r_switch.bound_switch(self = r_switch, comb = "33", zratio1 = numpy.array([[.3], [.8]]), zratio2 = numpy.array([[.3], [.8]])))
"""
"""print(ipol_11(numpy.column_stack((K, zratio1[0, : ], zratio2[0, : ]))))
print(r_switch.r_ml(self = r_switch, K = K, zratio1 = zratio1, zratio2 = zratio2, comb = "11"))
print(r_sol.batch(self = r_sol, K = K, comb = "11", zratio1 = zratio1, zratio2 = zratio2, tol = 1e-8))
print(K)
print(zratio1)
print(zratio2)
print(numpy.abs(K) > 1 * r_switch.bound_switch(self = r_switch, comb = "11", zratio1 = zratio1, zratio2 = zratio2))
print(r_switch.r_approx(self = r_switch, K = K, zratio1 = zratio1, zratio2 = zratio2, comb = "11", tol = 1e-8, ratio = 1))
print(r_switch.r_approx(self = r_switch, K = K, zratio1 = zratio1, zratio2 = zratio2, comb = "11", tol = 1e-8, ratio = 0))"""