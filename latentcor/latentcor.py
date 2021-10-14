
import internal
import numpy

def latentcor(X, tps, method, nu, tol, ratio):
    """Check the supplied parameters are compatible with what is expected."""
    if nu < 0 | nu > 1:
        print("nu must be between 0 and 1.")
        exit()
    elif tol <= 0:
        print("tol for optimization should be a positive value.")
        exit()
    elif ratio < 0 | ratio > 1:
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
    

    
