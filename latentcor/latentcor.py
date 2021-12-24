
import internal
import gen_data
import get_tps
import numpy
from statsmodels.stats.correlation_tools import corr_nearest
import seaborn
from matplotlib import pyplot
import pyreadr
import os

result = pyreadr.read_r(os.path.join(os.getcwd(), "latentcor", "data", "amgutpruned.rdata"))
ampdata = result['amgutpruned']
ampdata_mat = numpy.array(ampdata)
print(ampdata_mat)
print(numpy.sum(ampdata_mat==0, axis = 0))
print(len(numpy.sum(ampdata_mat==0, axis = 0)))
print(ampdata_mat.shape[1])

def latentcor(X, tps = None, method = "approx", use_nearPD = True, nu = 0.001, tol = 1e-8, ratio = 0.9, showplot = False):
    """Estimate latent correlation for mixed types.

    Estimation of latent correlation matrix from observed data of (possibly) mixed types (continuous/binary/truncated/ternary) based on the latent Gaussian copula model. Missing values (NA) are allowed. The estimation is based on pairwise complete observations.
    
    Parameters
    ----------

    X : numpy.array or pandas.DataFrame
        A numeric matrix or numeric data frame (n by p), where n is number of samples, and p is number of variables. Missing values (NA) are allowed, in which case the estimation is based on pairwise complete observations.
    
    tps : list or 1D-array
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
    K_a_lower = internal.Kendalltau.Kendalltau(self = internal.Kendalltau, X = X)
    zratios = internal.zratios.batch(self = internal.zratios, X = X, tps = tps)
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
                R_lower[comb_select] = internal.r_sol.batch(self = internal.r_sol, K = K, comb = comb, zratio1 = zratio1, zratio2 = zratio2, tol = tol)
            elif method == "approx":
                R_lower[comb_select] = internal.r_switch.r_approx(self = internal.r_switch, K = K, zratio1 = zratio1, zratio2 = zratio2, comb = comb, tol = tol, ratio  = ratio)
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

"""print(latentcor(X = [[1,2], [3,4]], tps = ["bin", "tru"], method = "original", use_nearPD = False, nu = -1, tol = .0001, ratio = .5))
"""
"""print(latentcor(X = [[1,2], [3,4]], tps = ["bin", "tru"], method = "original", use_nearPD = False, nu = .1, tol = 0, ratio = .5))    
"""
"""print(latentcor(X = [[1,2], [3,4]], tps = ["bin", "tru"], method = "original", use_nearPD = False, nu = .1, tol = 0.001, ratio = -1))"""
"""X = gen_data.gen_data(n = 100, rhos = .5, copulas = ["no"], tps = ["con", "bin", "tru", "ter"], XP = None)
print(X[0].shape[1])"""
 

"""print(latentcor(X = X[0], tps = ["con", "bin", "tru"], method = "original", use_nearPD = False, nu = .1, tol = .001, ratio = .5)[0])
"""
"""print(latentcor(X = X[0], tps = ["con", "bin", "tru", "ter"], method = "approx", use_nearPD = False, nu = .1, tol = .001, ratio = .5)[0])
"""


"""X = ampdata_mat[ : , 0:3]; tps = ["tru"] * 3; method = "approx"

X = gen_data.gen_data(n = 100, rhos = .5, copulas = ["no"], tps = ["con", "bin", "tru"], XP = None)
print(X[0].shape[1])
tps = ["con", "bin", "tru"]; method = "approx"   
use_nearPD = False; nu = 0.001; tol = 1e-5; ratio = 0.9; showplot = False

X = X[0]

X = numpy.array(X, dtype = float); nu = float(nu); tol = float(tol); ratio = float(ratio)
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

p = X.shape[1]
R = numpy.zeros((p, p), dtype = float)
cp = numpy.tril_indices(p, -1); cp_col = len(cp[1])
K_a_lower = internal.Kendalltau.Kendalltau(self = internal.Kendalltau, X = X)
zratios = internal.zratios.batch(self = internal.zratios, X = X, tps = tps)
tps_code = numpy.zeros(p, dtype = numpy.int32); tps_code[tps == "bin"] = 1
tps_code[tps == "tru"] = 2; tps_code[tps == "ter"] = 3
tps_cp = numpy.zeros((2, cp_col), dtype = numpy.int32)
tps_cp[0 , : ] = tps_code[cp[0]]; tps_cp[1, : ] = tps_code[cp[1]]
zratios_cp_0 = zratios[ : , cp[0]]; zratios_cp_1 = zratios[ : , cp[1]]
tps_mirror = tps_cp[0, : ] < tps_cp[1, : ]
tps_cp[ : , tps_mirror] = numpy.row_stack((tps_cp[1, tps_mirror], tps_cp[0, tps_mirror]))
zratios_cp_0_mirror = zratios_cp_0[ :, tps_mirror]; zratios_cp_1_mirror = zratios_cp_1[ :, tps_mirror]
zratios_cp_1[ : , tps_mirror] = zratios_cp_0_mirror; zratios_cp_0[ :, tps_mirror] = zratios_cp_1_mirror
combs_cp = numpy.repeat("NA", cp_col)
combs_cp = numpy.core.defchararray.add(numpy.array(tps_cp[0, : ], dtype = str), numpy.array(tps_cp[1, : ], dtype = str))
combs = numpy.unique(combs_cp)
R_lower = numpy.repeat(numpy.nan, cp_col)
print(zratios)
print(combs); print(zratios_cp_0); print(zratios_cp_1)
for comb in combs:
    comb_select = combs_cp == comb
    if comb == "00":
        R_lower[comb_select] = numpy.sin((numpy.pi / 2) * K_a_lower)
    else:
        K = K_a_lower[comb_select]
        zratio1 = zratios_cp_0[ :, comb_select]; zratio2 = zratios_cp_1[ :, comb_select]
        if method == "original":
            R_lower[comb_select] = internal.r_sol.batch(self = internal.r_sol, K = K, comb = comb, zratio1 = zratio1, zratio2 = zratio2, tol = tol)
        elif method == "approx":
            R_lower[comb_select] = internal.r_switch.r_approx(self = internal.r_switch, K = K, zratio1 = zratio1, zratio2 = zratio2, comb = comb, tol = tol, ratio  = ratio)
K = numpy.zeros((p, p), dtype = float)
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
print(K); print(R); print(zratios)

ampdata_approx = latentcor(X = ampdata_mat[ : , 0:3], tps = ["tru"] * 3, method = "approx")
print(ampdata_approx)

X = gen_data.gen_data(n = 100, rhos = .5, copulas = ["no"], tps = ["con", "bin", "tru"], XP = None)
print(latentcor(X = X[0], tps = ["con", "bin", "tru"], method = "approx", use_nearPD = False))"""