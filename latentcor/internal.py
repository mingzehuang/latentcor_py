"""Main module."""
import numpy
from scipy import stats
from scipy.optimize import minimize_scalar
from scipy.interpolate import interpn
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
    """Switch between different data types""" 
    def type_switch(self, type, copula, z, xp):
        method_name = type
        method = getattr(self, method_name, lambda: 'Invalid type')
        return method(self = fromZtoX, u = fromZtoX.copula_switch(self = fromZtoX, copula = copula, z = z), xp = xp)
    """Define continuous data"""
    def con (self, u, xp):
        return u
    """Define binary data"""
    def bin (self, u, xp):
        q = numpy.quantile(u, xp); x = numpy.zeros(len(u)); x[u > q] = 1
        return x
    """Define truncated data"""
    def tru (self, u, xp):
        q = numpy.quantile(u, xp); x = numpy.zeros(len(u)); x[u > q] = u - q; x[u <= numpy.repeat(q, len(u))] = 0
        return x
    """Define ternary data"""
    def ter (self, u, xp):
        q = numpy.quantile(u, numpy.cumsum(xp)); x = numpy.ones(len(u)); x[u > numpy.repeat(q[1], len(u))] = 2; x[u <= numpy.repeat(q[0], len(u))] = 0
        return x
"""Test class fromZtoX"""
print(fromZtoX.type_switch(self = fromZtoX, type = "ter", copula = "cube", z = numpy.random.standard_normal(100), xp = [0.3, 0.5]))

"""Calculate ties for variable"""
def n_x(x, n):
    x_info = numpy.unique(x, return_counts = True)
    if (len(x_info[0]) != n):
        x_counts = x_info[1]; t_x = x_counts[x_counts > 1]; out = numpy.sum(t_x * (t_x - 1) / 2)
    else:
        out = 0
    return out
"""Test function n_x"""
print(n_x(x = [1, 3, 4, 5, 6, 7, 3, 2], n = 8))

"""Calculate zratios for X"""
class zratios(object):
    def zratios_switch(self, X, type):
        method_name = type
        method = getattr(self, method_name, lambda: 'Invalid type')
        return method(self = zratios, X = X)
    def con(self, X):
        out = numpy.empty(X.shape[1])
        return out
    def bin(self, X):
        out = (X == 0).mean(axis = 1)
        return out
    def tru(self, X):
        out = (X == 0).mean(axis = 1)
    def ter(self, X):
        out = numpy.vstack((X == 0).mean(axis = 1), numpy.ones(X.shape[1]) - (X == 2).mean(axis = 1))
        return out
    def batch(self, X, types):
        out = numpy.empty(X.shape[1])
        for type in numpy.unique(types):
            out[types == type] = zratios.zratios_switch(self = zratios, X = X, type = type)
        return out

class r_sol(object):
    def bridge_switch(self, comb, r, zratio1, zratio2):
        method_name = comb
        method = getattr(self, "bridge_" + str(method_name), lambda: 'Invalid mixed types')
        return method(self = r_sol, r = r, zratio1 = zratio1, zratio2 = zratio2)
    def bridge_10(self, r, zratio1, zratio2):
        de1 = stats.norm.ppf(zratio1)
        res = 4 * stats.multivariate_normal.pdf(x = [de1, 0], cov = numpy.array([1, r / numpy.sqrt(2) - 2 * zratio1], [r / numpy.sqrt(2) - 2 * zratio1, 1]))
        return res
    def bridge_11(self, r, zratio1, zratio2):
        de1 = stats.norm.ppf(zratio1); de2 = stats.norm.ppf(zratio2)
        res = 2 * stats.multivariate_normal.pdf(x = [de1, de2], cov = numpy.array([1, r], [r, 1]) - zratio1 * zratio2)
        return res
    def bridge_20(self, r, zratio1, zratio2):
        de1 = stats.norm.ppf(zratio1)
        mat2 = numpy.array([1, 1 / numpy.sqrt(2), r / numpy.sqrt(2)], [1 / numpy.sqrt(2), 1, r], [r / numpy.sqrt(2), r, 1])
        res = - 2 * stats.multivariate_normal.pdf(x = [- de1, 0], cov = numpy.array([1, 1 / numpy.sqrt(2)], [1 / numpy.sqrt(2), 1])) \
              + 4 * stats.multivariate_normal.pdf(x = [- de1, 0, 0], cov = mat2)
        return res
    def bridge_21(self, r, zratio1, zratio2):
        de1 = stats.norm.ppf(zratio1); de2 = stats.norm.ppf(zratio2)
        mat1 = numpy.array([1, - r, 1 / numpy.sqrt(2)], [- r, 1, - r / numpy.sqrt(2)], [1 / numpy.sqrt(2), - r / numpy.sqrt(2), 1])
        mat2 = numpy.array([1, 0, - 1 / numpy.sqrt(2)], [0, 1, - r / numpy.sqrt(2)], [- 1 / numpy.sqrt(2), - r / numpy.sqrt(2), 1])
        res = 2 * (1 - zratio1) * zratio2 - 2 * stats.multivariate_normal.pdf(x = [- de1, de2, 0], cov = mat1) - 2 * stats.multivariate_normal.pdf(x = [-de1, de2, 0], cov = mat2)
        return res
    def bridge_22(self, r, zratio1, zratio2):
        de1 = stats.norm.ppf(zratio1); de2 = stats.norm.ppf(zratio2)
        mat1 = numpy.array([1, 0 , 1 / numpy.sqrt(2), - r / numpy.sqrt(2)], [0, 1, - r / numpy.sqrt(2), 1 / numpy.sqrt(2)], \
               [1 / numpy.sqrt(2), - r / numpy.sqrt(2), 1, - r], [- r / numpy.sqrt(2), 1 / numpy.sqrt(2), - r, 1])
        mat2 = numpy.array([1, r, 1 / numpy.sqrt(2), r / numpy.sqrt(2)], [r, 1, r / numpy.sqrt(2), 1 / numpy.sqrt(2)], \
               [1 / numpy.sqrt(2), r /numpy.sqrt(2), 1, r], [r / numpy.sqrt(2), 1 / numpy.sqrt(2), r, 1])
        res = - 2 * stats.multivariate_normal.pdf(x = [- de1, - de2, 0, 0], cov = mat1) + 2 * stats.multivariate_normal.pdf(x = [- de1, - de2, 0, 0], cov = mat2)
        return res
    def bridge_30(self, r, zratio1, zratio2):
        de1 = stats.norm.ppf(zratio1)
        mat = numpy.array([1, 0, r / numpy.sqrt(2)], [0, 1, - r / numpy.sqrt(2)], [r / numpy.sqrt(2), - r / numpy.sqrt(2), 1])
        res = 4 * stats.multivariate_normal.pdf(x = [de1[1], 0], cov = numpy.array([1, r / numpy.sqrt(2)], [r / numpy.sqrt(2), 1])) - 2 * zratio1[1] \
              + 4 * stats.multivariate_normal.pdf(x = [de1[0], de1[1], 0], cov = mat) - 2 * zratio1[0] * zratio1[1]
        return res
    def bridge_31(self, r, zratio1, zratio2):
        de1 = stats.norm.ppf(zratio1); de2 = stats.norm.ppf(zratio2)
        res = 2 * stats.multivariate_normal.pdf(x = [de2, de1[1]], cov = numpy.array([1, r], [r, 1])) * (1 - zratio1[0]) \
              - 2 * zratio1[1] * (zratio2 - stats.multivariate_normal.pdf(x = [de2, de1[0]], cov = numpy.array([1, r], [r, 1])))
        return res
    def bridge_32(self, r, zratio1, zratio2):
        de1 = stats.norm.ppf(zratio1); de2 = stats.norm.ppf(zratio2)
        mat1 = numpy.array([1, 0, 0], [0, 1, r], [0, r, 1])
        mat2 = numpy.array([1, 0, 0, r / numpy.sqrt(2)], [0, 1, - r, r / numpy.sqrt(2)], [0, - r, 1, - 1 / numpy.sqrt(2)], [r / numpy.sqrt(2), r / numpy.sqrt(2), - 1 / numpy.sqrt(2), 1])
        mat3 = numpy.array([1, 0, r, r / numpy.sqrt(2)], [0, 1, 0, r / numpy.sqrt(2)], [r, 0, 1, 1 / numpy.sqrt(2)], [r / numpy.sqrt(2), r / numpy.sqrt(2), 1 / numpy.sqrt(2), 1])
        res = - 2 * (1 - zratio1[0]) * zratio1[1] + 2 * stats.multivariate_normal.pdf(x = [- de1[0], de1[1], de2], cov = mat1) \
              + 2 * stats.multivariate_normal.pdf(x = [- de1[0], de1[1], - de2, 0], cov = mat2) + 2 * stats.multivariate_normal.pdf(x = [- de1[0], de1[1], - de2, 0], cov = mat3)
        return res
    def bridge_33(self, r, zratio1, zratio2):
        de1 = stats.norm.ppf(zratio1); de2 = stats.norm.ppf(zratio2)
        res = 2 * stats.multivariate_normal.pdf(x = [de1[1], de2[1]], cov = numpy.array([1, r], [r, 1])) * stats.multivariate_normal.pdf(x = [- de1[0], - de2[0]], cov = numpy.array([1, r], [r, 1]))
        return res
    def obj(self, r, K, comb, zratio1, zratio2):
        return (r_sol.bridge_switch(self = r_sol, r = r, comb = comb, zratio1 = zratio1, zratio2 = zratio2) - K) ** 2
    def batch(self, K, comb, zratio1, zratio2, tol):
        K_len = len(K); out = numpy.empty(K_len)
        for i in range(K_len):
            res = minimize_scalar(fun = r_sol.obj, args = [K[i], comb, zratio1[ : , i], zratio2[ : , i]], bounds = (-0.999, 0.999), method = 'bounded', tol = tol)
            res[i] = res[res.success == True].x
        return out

class r_switch(object):
    def bound_switch(self, comb, zratio1, zratio2):
        method_name = comb
        method = getattr(self, "bound_" + str(method_name), lambda: 'Invalid mixed types')
        return method(self = r_switch, zratio1 = zratio1, zratio2 = zratio2)
    def bound_10(self, zratio1, zratio2):
        return 2 * zratio1 * (1 - zratio1)
    def bound_11(self, zratio1, zratio2):
        return 2 * min(zratio1, zratio2) * (1 - max(zratio1, zratio2))
    def bound_20(self, zratio1, zratio2):
        return 1 - zratio1 ** 2
    def bound_21(self, zratio1, zratio2):
        return 2 * max(zratio2, 1 - zratio2) * (1 - max(zratio2, 1 - zratio2, zratio1))
    def bound_22(self, zratio1, zratio2):
        return 1 - max(zratio1, zratio2) ** 2
    def bound_30(self, zratio1, zratio2):
        return 2 * (zratio1[0] * (1 - zratio1[0]) + (1 - zratio1[1]) * (zratio1[1] - zratio1[0]))
    def bound_31(self, zratio1, zratio2):
        return 2 * min(zratio1[0] * (1 - zratio1[0]) + (1 - zratio1[1]) * (zratio1[1] - zratio1[0]), zratio2 * (1 - zratio2))
    def bound_32(self, zratio1, zratio2):
        return 1 - max(zratio1[0], zratio1[1] - zratio1[0], 1 - zratio1[1], zratio2) ** 2
    def bound_33(self, zratio1, zratio2):
        return 2 * min(zratio1[0] * (1 - zratio1[0]) + (1 - zratio1[1]) * (zratio1[1] - zratio1[0]), \
                       zratio2[0] * (1 - zratio2[0]) + (1 - zratio2[1]) * (zratio2[1] - zratio2[0]))
    def ipol_switch(self, comb, K, zratio1, zratio2):
        method_name = comb
        method = getattr(self, "ipol_" + str(method_name), lambda: 'Invalid mixed types')
        return method(self = r_switch, K = K, zratio1 = zratio1, zratio2 = zratio2)
    def r_ml(self, K, zratio1, zratio2, comb, ratio):
        zratio1_nrow = zratio1.shape[0]; zratio2_nrow = zratio2.shape[0]
        if zratio1_nrow > 1:
            zratio1[0:(zratio1_nrow - 2), ] = zratio1[0:(zratio1_nrow - 2), ] / zratio1[1:(zratio1_nrow - 1), ]
        if zratio2_nrow > 1:
            zratio2[0:(zratio2_nrow - 2), ] = zratio2[0:(zratio2_nrow - 2), ] / zratio2[1:(zratio2_nrow - 1), ]
        out = r_switch.ipol_switch(self = r_switch, K = K, zratio1 = zratio1, zratio2 = zratio2)
    def r_approx(self, K, zratio1, zratio2, comb, tol, ratio):
        bound = r_switch.bound_switch(self = r_switch, comb = comb, zratio1 = zratio1, zratio2 = zratio2)
        cutoff = numpy.abs(K) > ratio * bound
        if not any(cutoff):
            out = r_switch.r_ml(self = r_switch, K = K / bound, zratio1 = zratio1, zratio2 = zratio2, comb = comb, tol = tol, ratio = ratio)
        elif all(cutoff):
            out = r_sol.batch(K = K, zratio1 = zratio1, zratio2 = zratio2, comb = comb, tol = tol, ratio = ratio)
        else:
            out = numpy.empty(len(K)); revcutoff = numpy.logical_not(cutoff)
            out[cutoff] = r_sol.batch(self = r_sol, K = K[cutoff], zratio1 = zratio1[ : , cutoff], zratio2 = zratio2[ : , cutoff], comb = comb, tol = tol, ratio = ratio)
            out[revcutoff] = r_switch.r_ml(K = K[revcutoff] / bound[revcutoff], zratio1 = zratio1[ : , revcutoff], zratio2 = zratio2[ : , revcutoff], comb = comb, ratio = ratio)
        return out



