import numpy
def GenData(n, rhos, copulas, types, XP):
    if (len(n) != 1 | n <= 0):
        print("n should be a positive integer as sample size.")
        exit()
    p = len(types); p_copulas = len(copulas)
    if (p_copulas == 1):
        copulas = numpy.repeat(copulas, p)
    elif (p_copulas != p):
        print("copulas should have the same length as types, so that each copula corresponds to a variable (feature).")
    if (XP is None):
        XP = numpy.empty(p)
        for type in numpy.unique(types):
            XP[types == type] = getattr(type, lambda: 'Invalid type')