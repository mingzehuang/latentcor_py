import numpy


def get_tps(X, tru_prop = 0.5):
    X = numpy.array(X, dtype = float, ndmin = 2)
    p = X.shape[1]; tps = numpy.repeat("NA", p)
    for i in range(p):
        x = X[ : , i]
        x = x[numpy.logical_not(numpy.isnan(x))]
        levels = numpy.unique(x)
        if (len(levels) <= 1):
            print("No variation in " + str(i) + "th variable (" + str(i) + "th column of input data).")
            exit()
        elif (len(levels == 2)):
            """Two levels means binary"""
            tps[i] = "bin"
        elif (len(levels == 3)):
            """Three levels means ternary"""
            tps[i] = "ter"
        elif (len(levels > 3)):
            """More than 3 levels are detected - could be truncated or continuous"""
            if (len(levels) < 10):
                print("ordinal levels between 4 and 10 will be approximated by either countinuous or truncated type.")
            if (min(x) == 0) and (numpy.mean(x == 0) > tru_prop):
                """If the minimal value is zero and there are at least tru_prop zeros -> truncated (zero-inflated)"""
                tps[i] = "tru"
            else:
                tps[i] = "con"
    return tps
