import numpy
import internal
import gen_data

def get_tps(X, tru_prop = 0.05):
    """

    Parameters
    ----------

    X : numpy.array
        A numeric data matrix (n by p), where n is number of samples, and p is number of variables. Missing values (numpy.na) are allowed.
        
    tru_prop : float
               (Default value = 0.05)
               A scalar between 0 and 1 indicating the minimal proportion of zeros that should be present in a variable to be treated as `"tru"` (truncated type or zero-inflated) rather than as `"con"` (continuous type). The default value 0.05 means any variable with more than 5\% of zero values among n samples is treated as truncated or zero-inflated.

    Returns
    -------

    tps: numpy.array
         A vector of length p indicating the type of each of the p variables in `X`. Each element is one of `"con"` (continuous), `"bin"` (binary), `"ter"` (ternary) or `"tru"` (truncated).

    """

    X = numpy.array(X, dtype = float, ndmin = 2)
    p = X.shape[1]; tps = numpy.repeat("NAN", p)
    for i in range(p):
        x = X[ : , i]
        x = x[numpy.logical_not(numpy.isnan(x))]
        levels = numpy.unique(x)
        if (len(levels) <= 1):
            print("No variation in " + str(i) + "th variable (" + str(i) + "th column of input data).")
            exit()
        elif (len(levels) == 2):
            """Two levels means binary"""
            tps[i] = "bin"
        elif (len(levels) == 3):
            """Three levels means ternary"""
            tps[i] = "ter"
        elif (len(levels) > 3):
            """More than 3 levels are detected - could be truncated or continuous"""
            if (len(levels) < 10):
                print("ordinal levels between 4 and 10 will be approximated by either countinuous or truncated type.")
            if (min(x) == 0) and (numpy.mean(x == 0) > tru_prop):
                """If the minimal value is zero and there are at least tru_prop zeros -> truncated (zero-inflated)"""
                tps[i] = "tru"
            else:
                tps[i] = "con"
    return tps

X = gen_data.gen_data(n = 100, rhos = .5, copulas = "no", tps = ["con", "bin", "tru", "ter"], XP = None)[0]
print(X)
print(get_tps(X))
