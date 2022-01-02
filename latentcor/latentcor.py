
import os
import sys
sys.path.insert(0, os.path.abspath('../latentcor'))
import numpy
from statsmodels.stats.correlation_tools import corr_nearest
import seaborn
from matplotlib import pyplot
from scipy import stats
from scipy.optimize import fminbound
from joblib import Parallel, delayed
import pickle
import lzma

"""ipol_10_file = pkg_resources.resource_stream('data', 'ipol_10.xz')"""
with lzma.open(os.path.join(os.path.abspath('../latentcor'), "data", "ipol_10.xz"), "rb") as f:
    ipol_10 = pickle.load(f)

"""ipol_11_file = pkg_resources.resource_stream('data', 'ipol_11.xz')"""
with lzma.open(os.path.join(os.path.abspath('../latentcor'), "data", "ipol_11.xz"), "rb") as f:
    ipol_11 = pickle.load(f)

"""ipol_20_file = pkg_resources.resource_stream('data', 'ipol_20.xz')"""
with lzma.open(os.path.join(os.path.abspath('../latentcor'), "data", "ipol_20.xz"), "rb") as f:
    ipol_20 = pickle.load(f)

"""ipol_21_file = pkg_resources.resource_stream('data', 'ipol_21.xz')"""
with lzma.open(os.path.join(os.path.abspath('../latentcor'), "data", "ipol_21.xz"), "rb") as f:
    ipol_21 = pickle.load(f)

"""ipol_22_file = pkg_resources.resource_stream('data', 'ipol_22.xz')"""
with lzma.open(os.path.join(os.path.abspath('../latentcor'), "data", "ipol_22.xz"), "rb") as f:
    ipol_22 = pickle.load(f)

"""ipol_30_file = pkg_resources.resource_stream('data', 'ipol_30.xz')"""
with lzma.open(os.path.join(os.path.abspath('../latentcor'), "data", "ipol_30.xz"), "rb") as f:
    ipol_30 = pickle.load(f)

"""ipol_31_file = pkg_resources.resource_stream('data', 'ipol_31.xz')"""
with lzma.open(os.path.join(os.path.abspath('../latentcor'), "data", "ipol_31.xz"), "rb") as f:
    ipol_31 = pickle.load(f)

"""ipol_32_file = pkg_resources.resource_stream('data', 'ipol_32.xz')"""
with lzma.open(os.path.join(os.path.abspath('../latentcor'), "data", "ipol_32.xz"), "rb") as f:
    ipol_32 = pickle.load(f)

"""ipol_33_file = pkg_resources.resource_stream('data', 'ipol_33.xz')"""
with lzma.open(os.path.join(os.path.abspath('../latentcor'), "data", "ipol_33.xz"), "rb") as f:
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



def gen_data(n = 100, tps = ["ter", "con"], rhos = .5, copulas = "no", XP = None, showplot = False):
    """

    Parameters
    ----------
    n : int
        (Default value = 100)
        A positive integer indicating the sample size.

    tps : {list, numpy.array}
        (Default value = ["ter", "con"])
        A vector indicating the type of each variable, could be `"con"` (continuous), `"bin"` (binary), `"tru"` (truncated) or `"ter"` (ternary). The number of variables is determined based on the length of types, that is `p = length(tps)`. The default value `["ter", "con"]` which creates two variables: the first one is ternary, the second one is continuous.
        
    rhos : {list, numpy.array}
        (Default value = .5)
        A vector with lower-triangular elements of desired correlation matrix, e.g. `rhos = [.3, .5, .7]` means the correlation matrix is `[[1, .3, .5], [.3, 1, .7], [.5, .7, 1]]`. If only a scalar is supplied (`len(rhos) = 1`), then equi-correlation matrix is assumed with all pairwise correlations being equal to `rhos`. The default value is 0.5 which means correlations between any two variables are 0.5.

    copulas : {list, numpy.array}
        (Default value = "no")
        A vector indicating the copula transformation f for each of the p variables, e.g. U = f(Z). Each element can take value `"no"` (f is identity), `"expo"` (exponential transformation) or `"cube"` (cubic transformation). If the vector has length 1, then the same transformation is applied to all p variables. The default value is `"no"`: no copula transformation for any of the variables.

    XP : numpy.array
        (Default value = None)
        A (2 x p) matrix indicating proportion of zeros (for binary and truncated), and proportions of zeros and ones (for ternary) for each of the variables. For continuous variable, NA should be supplied. If `None`, the following values are automatically generated as elements of `XP` list for the corresponding data types:
        For continuous variable, the corresponding value is [numpy.na, numpy.na];
        for binary or truncated variable, the corresponding value is a number between 0 and 1 representing the proportion of zeros and numpy.na, the default value is [0.5, numpy.na];
        for ternary variable, the corresponding value is a pair of numbers between 0 and 1, the first number indicates the the proportion of zeros, the second number indicates the proportion of ones. The sum of a pair of numbers should be between 0 and 1, the default value is [0.3, 0.5].

    showplot : bool
        (Default value = False)
        If `True`, generates the plot of the data when number of variables p is no more than 3.

    Returns
    -------

    X : numpy.array
        Generated data matrix (n by p) of observed variables.
    
    plotX : object
        Visualization of the data matrix X.
        Histogram if `p=1`. 2D Scatter plot if `p=2`. 3D scatter plot if `p=3`. Returns None if `showplot = False`.

    """

    if (type(n) is not int) | (n <= 0):
        print("n should be a positive integer as sample size.")
        exit()
    n = int(n); rhos = float(rhos); copulas = numpy.array(copulas, dtype = str, ndmin = 1); tps = numpy.array(tps, dtype = str, ndmin = 1)
    p = len(tps); p_copulas = len(copulas)
    if p_copulas == 1:
        copulas = numpy.repeat(copulas, p)
    elif p_copulas != p:
        print("copulas should have the same length as tps, so that each copula corresponds to a variable (feature).")
    if XP is None:
        XP = numpy.repeat(numpy.NaN, 2 * p).reshape((2, p))
        XP[0, tps == "bin"] = .5; XP[0, tps == "tru"] = .5
        XP[0, tps == "ter"] = .3; XP[1, tps == "ter"] = .5
    elif XP.shape[1] != p:
        print("XP should be a list has the same length as tps, so that each element is a set of proportion(s) correponds to a variable (feature).")
        exit()
    elif ((numpy.sum(XP <= 0)) > 0) | ((numpy.sum(XP >= 1)) > 0):
        print("The proportion(s) should always between 0 and 1. Otherwise please consider to degenerate your data tp.")
        exit()
    XP = numpy.array(XP, dtype = float, ndmin = 2)
    if p == 1:
        Z = stats.norm.rvs(size = n).reshape((n, p))
    else:
        if len(copulas) == 1:
            copulas = numpy.repeat(copulas, p)
        lowertri = numpy.tril_indices(p, -1)
        rhos = numpy.array([rhos], dtype = float, ndmin = 1)
        if len(rhos) == 1:
            rhos = numpy.repeat(rhos, len(lowertri[1]))
        elif len(rhos) != len(lowertri[1]):
            print("Length of rhos should fit for lower triangular part of latent correlation matrix.")
            exit()
        Sigma_lower = numpy.zeros((p, p), dtype = float); Sigma_lower[lowertri] = rhos; Sigma = Sigma_lower + Sigma_lower.transpose()
        numpy.fill_diagonal(Sigma, 1)
        Z = stats.multivariate_normal.rvs(cov = Sigma, size = n)
    X = Z
    for i in range(p):
        X[ : , i] = fromZtoX.tp_switch(self = fromZtoX, tp = tps[i], copula = copulas[i], z = Z[ : , i], xp = XP[ : , i])
    plotX = None
    if (p == 1) & (showplot is True):
        plotX = seaborn.histplot(X)
        pyplot.show()
    elif (p == 2) & (showplot is True):
        plotX = seaborn.scatterplot(X)
        pyplot.show()   
    return X, plotX


def get_tps(X, tru_prop = 0.05):
    """

    Parameters
    ----------

    X : numpy.array
        A numeric data matrix (n by p), where n is number of samples, and p is number of variables. Missing values (numpy.na) are allowed.
        
    tru_prop : float
        (Default value = 0.05)
        A scalar between 0 and 1 indicating the minimal proportion of zeros that should be present in a variable to be treated as `"tru"` (truncated type or zero-inflated) rather than as `"con"` (continuous type). The default value 0.05 means any variable with more than 5% of zero values among n samples is treated as truncated or zero-inflated.

    Returns
    -------

    tps: numpy.array
        A vector of length p indicating the type of each of the p variables in `X`. Each element is one of `"con"` (continuous), `"bin"` (binary), `"ter"` (ternary) or `"tru"` (truncated).

    """

    X = numpy.array(X, dtype = float, ndmin = 2)
    p = X.shape[1]; tps = numpy.repeat("NAN", p)
    for i in range(p):
        x = X[ : , i]
        x = x[numpy.logical_not(numpy.isnan(x))]
        levels = numpy.unique(x)
        if (len(levels) <= 1):
            print("No variation in " + str(i) + "th variable (" + str(i) + "th column of input data).")
            exit()
        elif (len(levels) == 2):
            """Two levels means binary"""
            tps[i] = "bin"
        elif (len(levels) == 3):
            """Three levels means ternary"""
            tps[i] = "ter"
        elif (len(levels) > 3):
            """More than 3 levels are detected - could be truncated or continuous"""
            if (len(levels) < 10):
                print("ordinal levels between 4 and 10 will be approximated by either countinuous or truncated type.")
            if (min(x) == 0) and (numpy.mean(x == 0) > tru_prop):
                """If the minimal value is zero and there are at least tru_prop zeros -> truncated (zero-inflated)"""
                tps[i] = "tru"
            else:
                tps[i] = "con"
    return tps



def latentcor(X, tps = None, method = "approx", use_nearPD = True, nu = 0.001, tol = 1e-8, ratio = 0.9, showplot = False):
    """Estimate latent correlation for mixed types.

    Estimation of latent correlation matrix from observed data of (possibly) mixed types (continuous/binary/truncated/ternary) based on the latent Gaussian copula model. Missing values (NA) are allowed. The estimation is based on pairwise complete observations.
    
    Parameters
    ----------

    X : {numpy.array, pandas.DataFrame}
        A numeric matrix or numeric data frame (n by p), where n is number of samples, and p is number of variables. Missing values (NA) are allowed, in which case the estimation is based on pairwise complete observations.
    
    tps : {list, numpy.array}
        (Default value = None)
        A vector of length p indicating the type of each of the p variables in `X`. Each element must be one of "con" (continuous), "bin" (binary), "ter" (ternary) or "tru" (truncated). If the vector has length 1, then all p variables are assumed to be of the same type that is supplied. The variable types are determined automatically using function `get_tps`. As automatic determination of variable types takes extra time, it is recommended to supply the types explicitly when they are known in advance.
    
    method : {'original', 'approx'}
        (Default value = "approx")
        The calculation method for latent correlations. Either "original" or "approx". If `method = "approx"`, multilinear approximation method is used, which is much faster than the original method, see Yoon et al. (2021) for timing comparisons for various variable types. If `method = "original"`, optimization of the bridge inverse function is used.
    
    use_nearPD : bool
        (Default value = True)
        `use.nearPD = True` gets nearest positive definite matrix for the estimated latent correlation matrix with shrinkage adjustment by `nu`. Output `R` is the same as `Rpointwise` if `use.nearPD = False`.
    
    nu : float
        (Default value = 0.001)
        Shrinkage parameter for the correlation matrix, must be between 0 and 1. Guarantees that the minimal eigenvalue of returned correlation matrix is greater or equal to `nu`. When `nu = 0`, no shrinkage is performed, the returned correlation matrix will be semi-positive definite but not necessarily strictly positive definite. When `nu = 1`, the identity matrix is returned (not recommended).
    
    tol : float
        (Default value = 1e-8)
        When `method = "original"`, specifies the desired accuracy of the bridge function inversion via uniroot optimization and is passed to `optimize`. When `method = "approx"`, this parameter is ignored.
    
    ratio : float
        (Default value = 0.9)
        When `method = "approx"`, specifies the boundary value for multilinear interpolation, must be between 0 and 1. When `ratio = 0`, no linear interpolation is performed (the slowest execution) which is equivalent to `method = "original"`. When `ratio = 1`, linear interpolation is always performed (the fastest execution) but may lead to high approximation errors. The default (recommended) value controls the approximation error and has fast execution, see Yoon et al. (2021) for details. When `method = "original"`, this parameter is ignored.
    
    showplot : bool
        (Default value = False)
        `showplot = True` generates a seaborn object `plot` with the heatmap of latent correlation matrix `R`. `plot = None` if `showplot = False`.
    
    Returns
    -------

    R : numpy.array
        (p x p) Estimated latent correlation matrix for `X`.

    Rpointwise : numpy.array
        (p x p) Point-wise estimates of latent correlations for `X`. This matrix is not guaranteed to be semi-positive definite. This is the original estimated latent correlation matrix without adjustment for positive-definiteness.

    plot : {object, None}
        Heatmap plot of latent correlation matrix `R`, None if `showplot = FALSE`.

    K : numpy.array
        (p x p) Kendall Tau (Tau-a) Matrix for `X`. 

    zratios : numpy.array
        A (2 x p) matrix corresponding to each variable. Returns numpy.na for continuous variable; proportion of zeros for binary/truncated variables; the cumulative proportions of zeros and ones (e.g. first value is proportion of zeros, second value is proportion of zeros and ones) for ternary variable.

    """
    
    X = numpy.array(X, dtype = numpy.float32); nu = float(nu); tol = float(tol); ratio = float(ratio)
    """Check the supplied parameters are compatible with what is expected."""
    if (nu < 0) | (nu > 1):
        print("nu must be between 0 and 1.")
        exit()
    elif tol <= 0:
        print("tol for optimization should be a positive value.")
        exit()
    elif (ratio < 0) | (ratio > 1):
        print("ratio must be between 0 and 1.")
        exit()
    elif tps is None:
        tps = get_tps.get_tps(X)
    else:
        tps = numpy.array(tps, dtype = str, ndmin = 1)
    """Here I'll find some convenient way to check numeric matrix."""
    p = X.shape[1]
    R = numpy.zeros((p, p), dtype = numpy.float32)
    cp = numpy.tril_indices(p, -1); cp_col = len(cp[1])
    """Here I'll deal with NaN value."""
    K_a_lower = Kendalltau.Kendalltau(self = Kendalltau, X = X)
    zratios = zratio.batch(self = zratio, X = X, tps = tps)
    tps_code = numpy.zeros(p, dtype = numpy.int32); tps_code[tps == "bin"] = 1
    tps_code[tps == "tru"] = 2; tps_code[tps == "ter"] = 3
    tps_cp = numpy.zeros((2, cp_col), dtype = numpy.int32)   
    tps_cp[0 , : ] = tps_code[cp[0]]; tps_cp[1, : ] = tps_code[cp[1]]
    zratios_cp_0 = zratios[ : , cp[0]]; zratios_cp_1 = zratios[ : , cp[1]]
    tps_mirror = tps_cp[0, : ] < tps_cp[1, : ]
    tps_cp[ : , tps_mirror] = numpy.row_stack((tps_cp[1, tps_mirror], tps_cp[0, tps_mirror]))
    zratios_cp_0_mirror = zratios_cp_0[ :, tps_mirror]; zratios_cp_1_mirror = zratios_cp_1[ :, tps_mirror]
    zratios_cp_1[ :, tps_mirror] = zratios_cp_0_mirror; zratios_cp_0[ :, tps_mirror] = zratios_cp_1_mirror
    combs_cp = numpy.repeat("NA", cp_col)
    combs_cp = numpy.core.defchararray.add(numpy.array(tps_cp[0, : ], dtype = str), numpy.array(tps_cp[1, : ], dtype = str))
    combs = numpy.unique(combs_cp)
    R_lower = numpy.repeat(numpy.nan, cp_col)
    for comb in combs:
        comb_select = combs_cp == comb
        if comb == "00":
            R_lower[comb_select] = numpy.sin((numpy.pi / 2) * K_a_lower)
        else:
            K = K_a_lower[comb_select]
            """Here I need to deal with dimension degeneration"""
            zratio1 = zratios_cp_0[ :, comb_select]; zratio2 = zratios_cp_1[ :, comb_select]
            if method == "original":
                R_lower[comb_select] = r_sol.batch(self = r_sol, K = K, comb = comb, zratio1 = zratio1, zratio2 = zratio2, tol = tol)
            elif method == "approx":
                R_lower[comb_select] = r_switch.r_approx(self = r_switch, K = K, zratio1 = zratio1, zratio2 = zratio2, comb = comb, tol = tol, ratio  = ratio)
    K = numpy.zeros((p, p), dtype = numpy.float32)
    K[cp] = K_a_lower; R[cp] = R_lower
    K = K + K.transpose(); numpy.fill_diagonal(K, 1); R = R + R.transpose(); numpy.fill_diagonal(R, 1)
    Rpointwise = R
    if use_nearPD is True:
        R = corr_nearest(R)
        R = (1 - nu) * R; numpy.fill_diagonal(R, nu)
    plot = None
    if showplot is True:
        plot = seaborn.heatmap(R)
        pyplot.show()
    return R, Rpointwise, plot, K, zratios

if __name__ == '__main__':
    latentcor(sys.argv)


