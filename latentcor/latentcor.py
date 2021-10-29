
import internal
import gen_data
import numpy

def latentcor(X, tps, method, use_nearPD, nu, tol, ratio):
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
    """Here I'll find some convenient way to check numeric matrix."""
    p = X.shape(1)
    R = numpy.repeat(numpy.zeros, p * p).reshape(p, p)
    cp = numpy.tril_indices(p, -1); cp_col = cp.shape(1)
    """Here I'll deal with NaN value."""
    K_a_lower = internal.Kendalltau.Kendalltau(self = internal.Kendalltau, X = X)
    zratios = internal.zratios.batch(self = internal.zratios, X = X, tps = tps)
    tps_code = numpy.zeros(p); tps_code[tps == "bin"] = 1
    tps_code[tps == "tru"] = 2; tps_code[tps == "ter"] = 3
    tps_cp = zratios_cp = numpy.zeros(2 * cp_col).reshape(2, cp_col)
    for i in range(cp_col):
        tps_cp[ : , i] = tps_code[cp]; zratios_cp[ : , i] = zratios[cp]
    tps_mirror = tps_cp[0, : ] < tps_cp[1, : ]
    tps_cp[ : , tps_mirror] = numpy.row_stack((tps_cp[1, tps_mirror], tps_cp[0, tps_mirror]))
    zratios_cp[ : , tps_mirror] = numpy.row_stack((tps_cp[1, tps_mirror], tps_cp[0, tps_mirror]))
    combs_cp = numpy.repeat(numpy.NaN, cp_col)
    for i in range(cp_col):
        combs_cp[i] = str(tps_cp[0, i]) + str(tps_cp[1, i])
    combs = numpy.unique(combs_cp)
    R_lower = numpy.repeat(numpy.NaN, cp_col)
    for comb in combs:
        comb_select = combs_cp == comb
        if comb == "00":
            R_lower[comb_select] = numpy.sin((numpy.pi / 2) * K_a_lower)
        else:
            K = K_a_lower[comb_select]
            zratio1 = zratios_cp[1, comb_select]; zratio2 = zratios_cp[2, comb_select]
            if method == "original":
                R_lower[comb_select] = internal.r_sol.batch(self = internal.r_sol.batch, K = K, comb = comb, zratio1 = zratio1, zratio2 = zratio2, tol = tol)
            elif method == "approx":
                R_lower[comb_select] = internal.r_switch.r_approx(self = internal.r_switch.r_approx, K = K, zratio1 = zratio1, zratio2 = zratio2, comb = comb, tol = tol, ratio  = ratio)
    K = numpy.repeat(numpy.zeros, p * p).reshape(p, p)
    K[cp] = K_a_lower; K = K + K.T; numpy.fill_diagonal(K, 1)
    R[cp] = R + R.T; numpy.fill_diagonal(R, 1)
    Rpointwise = R
    if use_nearPD is True:
        R = (1 - nu) * R; numpy.fill_diagonal(R, nu)
    return zratios, K, R, Rpointwise

"""print(latentcor(X = [[1,2], [3,4]], tps = ["bin", "tru"], method = "original", use_nearPD = False, nu = -1, tol = .0001, ratio = .5))
"""
"""print(latentcor(X = [[1,2], [3,4]], tps = ["bin", "tru"], method = "original", use_nearPD = False, nu = .1, tol = 0, ratio = .5))    
"""
"""print(latentcor(X = [[1,2], [3,4]], tps = ["bin", "tru"], method = "original", use_nearPD = False, nu = .1, tol = 0.001, ratio = -1))"""
X = gen_data.gen_data(n = 100, rhos = .5, copulas = ["no"], tps = ["con", "bin", "tru", "ter"], XP = None)    
   
