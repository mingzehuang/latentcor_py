from latentcor import internal
import numpy
from scipy import stats
def GenData(n, rhos, copulas, types, XP):
    if len(n) != 1 | n <= 0:
        print("n should be a positive integer as sample size.")
        exit()
    p = len(types); p_copulas = len(copulas)
    if p_copulas == 1:
        copulas = numpy.repeat(copulas, p)
    elif p_copulas != p:
        print("copulas should have the same length as types, so that each copula corresponds to a variable (feature).")
    if XP is None:
        XP = numpy.empty(p)
        for type in numpy.unique(types):
            XP[types == type] = getattr("Data" + type, lambda: 'Invalid type')          
    elif type(XP) is not list | len(XP) != p:
        print("XP should be a list has the same length as types, so that each element is a set of proportion(s) correponds to a variable (feature).")
        exit()
    elif numpy.logical_not(XP > 0) | numpy.logical_not(XP < 1):
        print("The proportion(s) should always between 0 and 1. Otherwise please consider to degenerate your data type.")
        exit()
    if p == 1:
        Z = stats.norm.rvs(n)
    else:
        if len(copulas) == 1:
            copulas = numpy.repeat(copulas, p)
        lowertri = numpy.tril_indices(p)
        if len(rhos) != 1 & len(rhos) != len(lowertri):
            print("Length of rhos should fit for lower triangular part of latent correlation matrix.")
        Sigma_lower = numpy.zeros((p, p)); Sigma_lower[lowertri] = rhos; Sigma = Sigma_lower + Sigma_lower.transpose()
        numpy.fill_diagonal(Sigma, 1)
        Z = stats.multivariate_normal.rvs(cov = Sigma, size = n)
    X = numpy.empty(p)
    for i in range(p):
        X[i] = internal.fromZtoX.type_switch(self = internal.fromZtoX, type = types[i], copula = copulas[i], xp = XP[i])
    return X