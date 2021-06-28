"""Main module."""
import numpy
from scipy import stats
from scipy import optimize
class fromZtoX(object):
    def copula_switch(self, copula, z):
        method_name = copula
        method = getattr(self, method_name, lambda: 'Invalid copula')
        return method(self = fromZtoX, z = z)
    def no (self, z):
        return z
    def expo (self, z):
        return numpy.exp(z)
    def cube (self, z):
        return z ** 3
    def type_switch(self, type, copula, z, xp):
        method_name = type
        method = getattr(self, method_name, lambda: 'Invalid type')
        return method(self = fromZtoX, u = fromZtoX.copula_switch(self = fromZtoX, copula = copula, z = z), xp = xp)
    def con (self, u, xp):
        return u
    def bin (self, u, xp):
        q = numpy.quantile(u, xp); x = numpy.zeros(len(u)); x[u > q] = 1
        return x
    def tru (self, u, xp):
        q = numpy.quantile(u, xp); x = numpy.zeros(len(u)); x[u > q] = u - q; x[u <= numpy.repeat(q, len(u))] = 0
        return x
    def ter (self, u, xp):
        q = numpy.quantile(u, numpy.cumsum(xp)); x = numpy.ones(len(u)); x[u > numpy.repeat(q[1], len(u))] = 2; x[u <= numpy.repeat(q[0], len(u))] = 0
        return x
print(fromZtoX.type_switch(self = fromZtoX, type = "ter", copula = "cube", z = numpy.random.standard_normal(100), xp = [0.3, 0.5]))

def n_x(x, n):
    x_info = numpy.unique(x, return_counts = True)
    if (len(x_info[0]) != n):
        x_counts = x_info[1]; t_x = x_counts[x_counts > 1]; out = numpy.sum(t_x * (t_x - 1) / 2)
    else:
        out = 0
    return out
print(n_x(x = [1, 3, 4, 5, 6, 7, 3, 2], n = 8))

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
    def bridge_10(r, zratio1, zratio2):
        de1 = stats.norm.ppf(zratio1)
        res = 4 * stats.multivariate_normal.pdf(x = [de1, 0], cov = numpy.array([1, r / numpy.sqrt(2) - 2 * zratio1], [r / numpy.sqrt(2) - 2 * zratio1, 1]))
        return res
    def bridge_11(r, zratio1, zratio2):
        de1 = stats.norm.ppf(zratio1); de2 = stats.norm.ppf(zratio2)
        res = 2 * stats.multivariate_normal.pdf(x = [de1, de2], cov = numpy.array([1, r], [r, 1]) - zratio1 * zratio2)
        return res
    def bridge_20(r, zratio1, zratio2):
        de1 = stats.norm.ppf(zratio1)
        mat2 = numpy.array([1, 1 / numpy.sqrt(2), r / numpy.sqrt(2)], [1 / numpy.sqrt(2), 1, r], [r / numpy.sqrt(2), r, 1])
        res = - 2 * stats.multivariate_normal.pdf(x = [- de1, 0], cov = numpy.array([1, 1 / numpy.sqrt(2)], [1 / numpy.sqrt(2), 1])) \
              + 4 * stats.multivariate_normal.pdf(x = [- de1, 0, 0], cov = mat2)
        return res
    def bridge_21(r, zratio1, zratio2):
        de1 = stats.norm.ppf(zratio1); de2 = stats.norm.ppf(zratio2)
        mat1 = numpy.array([1, - r, 1 / numpy.sqrt(2)], [- r, 1, - r / numpy.sqrt(2)], [1 / numpy.sqrt(2), - r / numpy.sqrt(2), 1])
        mat2 = numpy.array([1, 0, - 1 / numpy.sqrt(2)], [0, 1, - r / numpy.sqrt(2)], [- 1 / numpy.sqrt(2), - r / numpy.sqrt(2), 1])
        res = 2 * (1 - zratio1) * zratio2 - 2 * stats.multivariate_normal.pdf(x = [- de1, de2, 0], cov = mat1) - 2 * stats.multivariate_normal.pdf(x = [-de1, de2, 0], cov = mat2)
        return res
    def bridge_22(r, zratio1, zratio2):
        de1 = stats.norm.ppf(zratio1); de2 = stats.norm.ppf(zratio2)
        mat1 = numpy.array([1, 0 , 1 / numpy.sqrt(2), - r / numpy.sqrt(2)], [0, 1, - r / numpy.sqrt(2), 1 / numpy.sqrt(2)], \
               [1 / numpy.sqrt(2), - r / numpy.sqrt(2), 1, - r], [- r / numpy.sqrt(2), 1 / numpy.sqrt(2), - r, 1])
        mat2 = numpy.array([1, r, 1 / numpy.sqrt(2), r / numpy.sqrt(2)], [r, 1, r / numpy.sqrt(2), 1 / numpy.sqrt(2)], \
               [1 / numpy.sqrt(2), r /numpy.sqrt(2), 1, r], [r / numpy.sqrt(2), 1 / numpy.sqrt(2), r, 1])
        res = - 2 * stats.multivariate_normal.pdf(x = [- de1, - de2, 0, 0], cov = mat1) + 2 * stats.multivariate_normal.pdf(x = [- de1, - de2, 0, 0], cov = mat2)
        return res
    def bridge_30(r, zratio1, zratio2):
        de1 = stats.norm.ppf(zratio1)
        mat = numpy.array([1, 0, r / numpy.sqrt(2)], [0, 1, - r / numpy.sqrt(2)], [r / numpy.sqrt(2), - r / numpy.sqrt(2), 1])
        res = 4 * stats.multivariate_normal.pdf(x = [de1[1], 0], cov = numpy.array([1, r / numpy.sqrt(2)], [r / numpy.sqrt(2), 1])) - 2 * zratio1[1] \
              + 4 * stats.multivariate_normal.pdf(x = [de1[0], de1[1], 0], cov = mat) - 2 * zratio1[0] * zratio1[1]
        return res
    def bridge_31(r, zratio1, zratio2):
        de1 = stats.norm.ppf(zratio1); de2 = stats.norm.ppf(zratio2)
        res = 2 * stats.multivariate_normal.pdf(x = [de2, de1[1]], cov = numpy.array([1, r], [r, 1])) * (1 - zratio1[0]) \
              - 2 * zratio1[1] * (zratio2 - stats.multivariate_normal.pdf(x = [de2, de1[0]], cov = numpy.array([1, r], [r, 1])))
        return res
    def bridge_32(r, zratio1, zratio2):
        de1 = stats.norm.ppf(zratio1); de2 = stats.norm.ppf(zratio2)
        mat1 = numpy.array([1, 0, 0], [0, 1, r], [0, r, 1])
        mat2 = numpy.array([1, 0, 0, r / numpy.sqrt(2)], [0, 1, - r, r / numpy.sqrt(2)], [0, - r, 1, - 1 / numpy.sqrt(2)], [r / numpy.sqrt(2), r / numpy.sqrt(2), - 1 / numpy.sqrt(2), 1])
        mat3 = numpy.array([1, 0, r, r / numpy.sqrt(2)], [0, 1, 0, r / numpy.sqrt(2)], [r, 0, 1, 1 / numpy.sqrt(2)], [r / numpy.sqrt(2), r / numpy.sqrt(2), 1 / numpy.sqrt(2), 1])
        res = - 2 * (1 - zratio1[0]) * zratio1[1] + 2 * stats.multivariate_normal.pdf(x = [- de1[0], de1[1], de2], cov = mat1) \
              + 2 * stats.multivariate_normal.pdf(x = [- de1[0], de1[1], - de2, 0], cov = mat2) + 2 * stats.multivariate_normal.pdf(x = [- de1[0], de1[1], - de2, 0], cov = mat3)
        return res
    def bridge_33(r, zratio1, zratio2):
        de1 = stats.norm.ppf(zratio1); de2 = stats.norm.ppf(zratio2)
        res = 2 * stats.multivariate_normal.pdf(x = [de1[1], de2[1]], cov = numpy.array([1, r], [r, 1])) * stats.multivariate_normal.pdf(x = [- de1[0], - de2[0]], cov = numpy.array([1, r], [r, 1]))
        return res
    def obj(r, K, comb, zratio1, zratio2):
        return (r_sol.bridge_switch(self = r_sol, r = r, comb = comb, zratio1 = zratio1, zratio2 = zratio2) - K) ** 2
    def batch(K, comb, zratio1, zratio2, tol):
        K_len = len(K); out = numpy.empty(K_len)
        for i in range(K_len):
            res = optimize.minimize_scalar(fun = r_sol.obj, args = [K[i], comb, zratio1[ : , i], zratio2[ : , i]], bounds = (-0.999, 0.999), method = 'bounded', tol = tol)
            res[i] = res[res.success == True].x
        return out

class r_switch(object):
    def bound_switch(self, comb, zratio1, zratio2):
        method_name = comb
        method = getattr(self, "bound_" + str(method_name), lambda: 'Invalid mixed types')
        return method(self = r_switch, zratio1 = zratio1, zratio2 = zratio2)
    def bound_10(zratio1, zratio2)
