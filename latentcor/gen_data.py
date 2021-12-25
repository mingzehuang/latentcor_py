
import internal
import numpy
import seaborn
from matplotlib import pyplot
from scipy import stats

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
               If TRUE, generates the plot of the data when number of variables p is no more than 3.

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
        X[ : , i] = internal.fromZtoX.tp_switch(self = internal.fromZtoX, tp = tps[i], copula = copulas[i], z = Z[ : , i], xp = XP[ : , i])
    plotX = None
    if (p == 1) & (showplot is True):
        plotX = seaborn.histplot(X)
        pyplot.show()
    elif (p == 2) & (showplot is True):
        plotX = seaborn.scatterplot(X)
        pyplot.show()   
    return X, plotX
  

"""print(stats.multivariate_normal.rvs(cov = [[1,.5],[.5,1]], size = 100))
print(gen_data(n = 100, rhos = .5, copulas = ["no"], tps = ["con"], XP = None))
print(gen_data(n = 100, rhos = .5, copulas = ["no"], tps = ["bin"], XP = None))
print(gen_data(n = 100, rhos = .5, copulas = ["no"], tps = ["tru"], XP = None))
print(gen_data(n = 100, rhos = .5, copulas = ["no"], tps = ["ter"], XP = None))"""
"""a=numpy.array([[1,2],[3,4]])
print(a)
b=numpy.array(["con", "bin"])"""
"""print(len(b))
print(a[0,b=="bin"])
print(gen_data(n = 100, rhos = .5, copulas = ["no"], tps = ["con", "bin"], XP = None))
print(gen_data(n = 100, rhos = .5, copulas = ["no"], tps = ["bin", "tru"], XP = None))
print(gen_data(n = 100, rhos = .5, copulas = ["no"], tps = ["tru", "ter"], XP = None))"""
print(gen_data(n = 100, rhos = .5, copulas = ["no"], tps = ["con", "bin", "tru", "ter"], XP = None)[0])