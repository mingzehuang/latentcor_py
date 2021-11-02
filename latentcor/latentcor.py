
import internal
import gen_data
import get_tps
import numpy
from statsmodels.stats.correlation_tools import corr_nearest
import seaborn
from matplotlib import pyplot

def latentcor(X, tps = None, method = "approx", use_nearPD = True, nu = 0.001, tol = 1e-8, ratio = 0.5, showplot = False):
    X = numpy.array(X, dtype = float); nu = float(nu); tol = float(tol); ratio = float(ratio)
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
    R = numpy.zeros((p, p), dtype = float)
    cp = numpy.tril_indices(p, -1); cp_col = len(cp[1])
    """Here I'll deal with NaN value."""
    K_a_lower = internal.Kendalltau.Kendalltau(self = internal.Kendalltau, X = X)
    zratios = internal.zratios.batch(self = internal.zratios, X = X, tps = tps)
    tps_code = numpy.zeros(p, dtype = int); tps_code[tps == "bin"] = 1
    tps_code[tps == "tru"] = 2; tps_code[tps == "ter"] = 3
    tps_cp = numpy.zeros((2, cp_col), dtype = int)
    zratios_cp = numpy.zeros((2, 2, cp_col), dtype = float)
    tps_cp[0 , : ] = tps_code[cp[0]]; tps_cp[1, : ] = tps_code[cp[1]]
    zratios_cp[0, : , : ] = zratios[ : , cp[0]]; zratios_cp[1, : , : ] = zratios[ : , cp[1]]
    tps_mirror = tps_cp[0, : ] < tps_cp[1, : ]
    tps_cp[ : , tps_mirror] = numpy.row_stack((tps_cp[1, tps_mirror], tps_cp[0, tps_mirror]))
    zratios_cp_0_mirror = zratios_cp[0, :, tps_mirror]; zratios_cp_1_mirror = zratios_cp[1, :, tps_mirror]
    zratios_cp[1, :, tps_mirror] = zratios_cp_0_mirror; zratios_cp[0, :, tps_mirror] = zratios_cp_1_mirror
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
            zratio1 = zratios_cp[0, :, comb_select].reshape(2, len(K)); zratio2 = zratios_cp[1, :, comb_select].reshape(2, len(K))
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
    return R, Rpointwise, plot, K, zratios

"""print(latentcor(X = [[1,2], [3,4]], tps = ["bin", "tru"], method = "original", use_nearPD = False, nu = -1, tol = .0001, ratio = .5))
"""
"""print(latentcor(X = [[1,2], [3,4]], tps = ["bin", "tru"], method = "original", use_nearPD = False, nu = .1, tol = 0, ratio = .5))    
"""
"""print(latentcor(X = [[1,2], [3,4]], tps = ["bin", "tru"], method = "original", use_nearPD = False, nu = .1, tol = 0.001, ratio = -1))"""
X = gen_data.gen_data(n = 100, rhos = .5, copulas = ["no"], tps = ["con", "bin", "tru", "ter"], XP = None)
print(X[0].shape[1])    
latentcor(X = X[0], tps = ["con", "bin", "tru", "ter"], method = "original", use_nearPD = False, nu = .1, tol = .001, ratio = .5)