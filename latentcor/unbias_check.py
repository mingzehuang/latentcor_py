import internal
import latentcor
import get_tps
import gen_data
import numpy
import seaborn
from matplotlib import pyplot
from scipy import stats
import pyreadr
import os
from rpy2 import robjects
import pandas
import timeit
from numba import jit
import lzma
import pickle
import pkg_resources
import pandas

result = pyreadr.read_r(os.path.join(os.getcwd(), "latentcor", "data", "amgutpruned.rdata"))
ampdata = result['amgutpruned']
ampdata_mat = numpy.array(ampdata)
print(ampdata_mat)
print(numpy.sum(ampdata_mat==0, axis = 0))
print(len(numpy.sum(ampdata_mat==0, axis = 0)))
print(ampdata_mat.shape[1])

"""starttime = timeit.default_timer()
internal.Kendalltau.Kendalltau(self = internal.Kendalltau, X = ampdata_mat[ : , : ])
print(timeit.default_timer() - starttime)
starttime = timeit.default_timer()
ampdata_df = pandas.DataFrame(ampdata_mat[ : , : ])
ampdata_df.corr(method = "kendall")
print(timeit.default_timer() - starttime)"""

"""ampdata_org = latentcor.latentcor(X = ampdata_mat[ : , 0:10], tps = ["tru"] * 10, method = "original")"""
"""ampdata_approx = latentcor.latentcor(X = ampdata_mat, tps = ["tru"] * ampdata_mat.shape[1], method = "approx", use_nearPD = False)"""
"""print(ampdata_org)"""
"""print(ampdata_approx)"""
"""print(internal.Kendalltau.Kendalltau(self = internal.Kendalltau, X = ampdata_mat[:,0:100]))"""
"""pd_X = pandas.DataFrame(ampdata_mat[:,0:100])
print(pd_X.corr(method = 'kendall'))"""

import_module = '''
import internal
import latentcor
import get_tps
import gen_data
import numpy
import seaborn
from matplotlib import pyplot
from scipy import stats
import pyreadr
import os
from rpy2 import robjects
import pandas
import timeit
result = pyreadr.read_r(os.path.join(os.getcwd(), "latentcor", "data", "amgutpruned.rdata"))
ampdata = result['amgutpruned']
ampdata_mat = numpy.array(ampdata)
'''

amp_org_20 = '''
def amp_org_20():
    latentcor.latentcor(X = ampdata_mat[ : , 0:20], tps = ["tru"] * 20, method = "original")
'''
amp_approx_20 = '''
def amp_approx_20():
    latentcor.latentcor(X = ampdata_mat[ : , 0:20], tps = ["tru"] * 20, method = "approx")
'''

print("amp original p = 10: ", timeit.timeit(stmt = amp_org_20, setup = import_module))
print("amp approx p = 10: ", timeit.timeit(stmt = amp_approx_20, setup = import_module))


"""starttime = timeit.default_timer()
print("start time original p = 20 is: ", starttime)
latentcor.latentcor(X = ampdata_mat[ : , 0:20], tps = ["tru"] * 20, method = "original", use_nearPD=False)
print("time difference original p = 20 is: ", timeit.default_timer() - starttime)
starttime = timeit.default_timer()
print("start time approx p = 20 is: ", starttime)
latentcor.latentcor(X = ampdata_mat[ : , 0:20], tps = ["tru"] * 20, method = "approx", use_nearPD=False)
print("time difference approx p = 20 is: ", timeit.default_timer() - starttime)

starttime = timeit.default_timer()
print("start time original p = 50 is: ", starttime)
latentcor.latentcor(X = ampdata_mat[ : , 0:50], tps = ["tru"] * 50, method = "original", use_nearPD=False)
print("time difference original p = 50 is: ", timeit.default_timer() - starttime)
starttime = timeit.default_timer()
print("start time approx p = 50 is: ", starttime)
latentcor.latentcor(X = ampdata_mat[ : , 0:50], tps = ["tru"] * 50, method = "approx", use_nearPD=False)
print("time difference approx p = 50 is: ", timeit.default_timer() - starttime)"""


"""all_p = [10, 20, 50, 100, 200, 300, 400, 481]; all_n = [100, 6482]

timing = numpy.full((len(all_p), len(all_n) * 2), numpy.nan)
for j in range(len(all_p)):
    for i in range(len(all_n)):
        p = all_p[j]; n = all_n[i]
        print("p = ", p, "n = ", n)
        starttime = timeit.default_timer()
        latentcor.latentcor(X = ampdata_mat[0:n, 0:p], tps = ["tru"] * p, method = "original", use_nearPD=False, tol = 1e-5)
        timing[j, 2 * i] = timeit.default_timer() - starttime
        print(timing[j, 2 * i])
        starttime = timeit.default_timer()
        latentcor.latentcor(X = ampdata_mat[0:n, 0:p], tps = ["tru"] * p, method = "approx", use_nearPD=False, tol = 1e-5)
        timing[j, 2 * i + 1] = timeit.default_timer() - starttime
        print(timing[j, 2 * i + 1])

print(timing)"""
"""with lzma.open(os.path.join(os.getcwd(), "latentcor", "data", "timing_all"), "wb", preset = 9) as f:
    pickle.dump(timing, f)

timing_all = pkg_resources.resource_stream('data', 'timing_all')
with lzma.open(timing_all, "rb") as f:
    timing = pickle.load(f)
print(timing)

data = {"log10 of time (original n = 100)": numpy.log10(timing[ : , 0]), "log10 of time (approx n = 100)": numpy.log10(timing[ : , 1]), "log10 of time (original n = 6482)": numpy.log10(timing[ : , 2]), "log10 of time (approx n = 6482)": numpy.log10(timing[ : , 3])}
dfdata = pandas.DataFrame(data, index = numpy.log10(all_p))
print(dfdata)

plot = seaborn.lineplot(data = dfdata, marker="o")

pyplot.title("timing_all")
with lzma.open(os.path.join(os.getcwd(), "latentcor", "data", "timing_plot"), "wb", preset = 9) as f:
    pickle.dump(plot, f)
pyplot.show()"""



"""ampdata = robjects.r.load(os.path.join(os.getcwd(), "latentcor", "data", "amgutpruned.rdata"))
print(ampdata[0])"""


"""Need more check"""
"""Ternary vs. Binary"""
"""rhos = numpy.linspace(-1,1,100); rhorep = numpy.repeat(numpy.nan, len(rhos))
Rrep = numpy.repeat(numpy.nan, len(rhos) * 3).reshape(len(rhos), 3)
for r in range(len(rhos)):
    X = gen_data.gen_data(n = 1000, tps = ["ter", "bin"], rhos = rhos[r], XP = numpy.array([[.3, .5], [.5, numpy.nan]], dtype = float, ndmin = 2), showplot = False)[0]
    R_nc_org = latentcor.latentcor(X = X, tps = ["ter", "bin"], method = "original", use_nearPD = False)[1]
    R_nc_approx = latentcor.latentcor(X = X, tps = ["ter", "bin"], method = "approx", ratio = .9, use_nearPD = False)[1]
    rhorep[r] = rhos[r]; Rrep[r, 0] = R_nc_org[1, 0]; Rrep[r, 1] = R_nc_approx[1, 0]
print(Rrep)
print(rhorep)
data = {"True latent correlation": rhorep, "Estimated latent correlation (original)": Rrep[ : , 0]}
plot = seaborn.scatterplot(data = data, x = "True latent correlation", y = "Estimated latent correlation (original)")
pyplot.plot(rhos, rhos, color = "r")
pyplot.title("Ternary vs. Binary (original)")
pyplot.show()

data = {"True latent correlation": rhorep, "Estimated latent correlation (approx)": Rrep[ : , 1]}
plot = seaborn.scatterplot(data = data, x = "True latent correlation", y = "Estimated latent correlation (approx)")
pyplot.plot(rhos, rhos, color = "r")
pyplot.title("Ternary vs. Binary (approx)")
pyplot.show()"""

"""
""""""Need more check""""""
""""""Ternary vs. Truncated""""""
rhos = numpy.linspace(-1,1,100); rhorep = numpy.repeat(numpy.nan, len(rhos))
Rrep = numpy.repeat(numpy.nan, len(rhos) * 3).reshape(len(rhos), 3)
for r in range(len(rhos)):
    X = gen_data.gen_data(n = 100, tps = ["ter", "tru"], rhos = rhos[r], XP = numpy.array([[.3, .5], [.5, numpy.nan]], dtype = float, ndmin = 2), showplot = False)[0]
    R_nc_org = latentcor.latentcor(X = X, tps = ["ter", "tru"], method = "original", use_nearPD = False)[1]
    R_nc_approx = latentcor.latentcor(X = X, tps = ["ter", "tru"], method = "approx", use_nearPD = False)[1]
    rhorep[r] = rhos[r]; Rrep[r, 0] = R_nc_org[1, 0]; Rrep[r, 1] = R_nc_approx[1, 0]; Rrep[r, 2] = numpy.corrcoef(X)[0,1]
print(Rrep)
print(rhorep)
data = {"True latent correlation": rhorep, "Estimated latent correlation (original)": Rrep[ : , 0]}
plot = seaborn.scatterplot(data = data, x = "True latent correlation", y = "Estimated latent correlation (original)")
pyplot.plot(rhos, rhos, color = "r")
pyplot.title("Ternary vs. Truncated (original)")
pyplot.show()

data = {"True latent correlation": rhorep, "Estimated latent correlation (approx)": Rrep[ : , 1]}
plot = seaborn.scatterplot(data = data, x = "True latent correlation", y = "Estimated latent correlation (approx)")
pyplot.plot(rhos, rhos, color = "r")
pyplot.title("Ternary vs. Truncated (approx)")
pyplot.show()
"""

"""Ternary vs. Ternary"""
"""rhos = numpy.linspace(-1,1,100); rhorep = numpy.repeat(numpy.nan, len(rhos))
Rrep = numpy.repeat(numpy.nan, len(rhos) * 3).reshape(len(rhos), 3)
for r in range(len(rhos)):
    X = gen_data.gen_data(n = 100, tps = ["ter", "ter"], rhos = rhos[r], XP = numpy.array([[.3, .5], [.3, .5]], dtype = float, ndmin = 2), showplot = False)[0]
    R_nc_org = latentcor.latentcor(X = X, tps = ["ter", "ter"], method = "original", use_nearPD = False)[1]
    rhorep[r] = rhos[r]; Rrep[r, 0] = R_nc_org[1, 0]; Rrep[r, 2] = numpy.corrcoef(X)[0,1]
print(Rrep)
print(rhorep)
data = {"True latent correlation": rhorep, "Estimated latent correlation": Rrep[ : , 0]}
plot = seaborn.scatterplot(data = data, x = "True latent correlation", y = "Estimated latent correlation")
pyplot.plot(rhos, rhos, color = "r")
pyplot.title("Ternary vs. Ternary")
pyplot.show()"""

"""Binary vs. Ternary"""
"""rhos = numpy.linspace(-1,1,100); rhorep = numpy.repeat(numpy.nan, len(rhos))
Rrep = numpy.repeat(numpy.nan, len(rhos) * 3).reshape(len(rhos), 3)
for r in range(len(rhos)):
    X = gen_data.gen_data(n = 100, tps = ["bin", "ter"], rhos = rhos[r], XP = numpy.array([[.5, .3], [numpy.nan, .5]], dtype = float, ndmin = 2), showplot = False)[0]
    R_nc_org = latentcor.latentcor(X = X, tps = ["bin", "ter"], method = "original", use_nearPD = False)[1]
    R_nc_approx = latentcor.latentcor(X = X, tps = ["bin", "ter"], method = "approx", use_nearPD = False)[1]
    rhorep[r] = rhos[r]; Rrep[r, 0] = R_nc_org[1, 0]; Rrep[r, 1] = R_nc_approx[1, 0]; Rrep[r, 2] = numpy.corrcoef(X)[0,1]
print(Rrep)
print(rhorep)
data = {"True latent correlation": rhorep, "Estimated latent correlation (original)": Rrep[ : , 0]}
plot = seaborn.scatterplot(data = data, x = "True latent correlation", y = "Estimated latent correlation (original)")
pyplot.plot(rhos, rhos, color = "r")
pyplot.title("Binary vs. Ternary (original)")
pyplot.show()
data = {"True latent correlation": rhorep, "Estimated latent correlation (approx)": Rrep[ : , 1]}
plot = seaborn.scatterplot(data = data, x = "True latent correlation", y = "Estimated latent correlation (approx)")
pyplot.plot(rhos, rhos, color = "r")
pyplot.title("Binary vs. Ternary (approx)")
pyplot.show()"""

"""Truncated vs. Ternary"""
"""rhos = numpy.linspace(-1,1,100); rhorep = numpy.repeat(numpy.nan, len(rhos))
Rrep = numpy.repeat(numpy.nan, len(rhos) * 3).reshape(len(rhos), 3)
for r in range(len(rhos)):
    X = gen_data.gen_data(n = 100, tps = ["tru", "ter"], rhos = rhos[r], XP = numpy.array([[.5, .3], [numpy.nan, .5]], dtype = float, ndmin = 2), showplot = False)[0]
    R_nc_org = latentcor.latentcor(X = X, tps = ["tru", "ter"], method = "original", use_nearPD = False)[1]
    rhorep[r] = rhos[r]; Rrep[r, 0] = R_nc_org[1, 0]; Rrep[r, 2] = numpy.corrcoef(X)[0,1]
print(Rrep)
print(rhorep)
data = {"True latent correlation": rhorep, "Estimated latent correlation": Rrep[ : , 0]}
plot = seaborn.scatterplot(data = data, x = "True latent correlation", y = "Estimated latent correlation")
pyplot.plot(rhos, rhos, color = "r")
pyplot.title("Truncated vs. Ternary")
pyplot.show()"""

"""Continuous vs. Ternary"""
"""rhos = numpy.linspace(-1,1,100); rhorep = numpy.repeat(numpy.nan, len(rhos))
Rrep = numpy.repeat(numpy.nan, len(rhos) * 3).reshape(len(rhos), 3)
for r in range(len(rhos)):
    X = gen_data.gen_data(n = 100, tps = ["con", "ter"], rhos = rhos[r], XP = numpy.array([[numpy.nan, .3], [numpy.nan, .5]], dtype = float, ndmin = 2), showplot = False)[0]
    R_nc_org = latentcor.latentcor(X = X, tps = ["con", "ter"], method = "original", use_nearPD = False)[1]
    R_nc_approx = latentcor.latentcor(X = X, tps = ["con", "ter"], method = "approx", use_nearPD = False)[1]
    rhorep[r] = rhos[r]; Rrep[r, 0] = R_nc_org[1, 0]; Rrep[r, 1] = R_nc_approx[1, 0]; Rrep[r, 2] = numpy.corrcoef(X)[0,1]
print(Rrep)
print(rhorep)
data = {"True latent correlation": rhorep, "Estimated latent correlation (original)": Rrep[ : , 0]}
plot = seaborn.scatterplot(data = data, x = "True latent correlation", y = "Estimated latent correlation (original)")
pyplot.plot(rhos, rhos, color = "r")
pyplot.title("Continuous vs. Ternary (original)")
pyplot.show()
data = {"True latent correlation": rhorep, "Estimated latent correlation (approx)": Rrep[ : , 1]}
plot = seaborn.scatterplot(data = data, x = "True latent correlation", y = "Estimated latent correlation (approx)")
pyplot.plot(rhos, rhos, color = "r")
pyplot.title("Continuous vs. Ternary (approx)")
pyplot.show()"""


"""All Combination"""

"""rhos = numpy.linspace(-1,1,100); rhorep = numpy.repeat(numpy.nan, len(rhos))
Rrep = numpy.full((len(rhos), 2), numpy.nan)
tps = ["con", "bin", "tru", "ter"]
for tp1 in range(4):
    for tp2 in range(4):
        tp_comb = [tps[tp1], tps[tp2]]
        for r in range(len(rhos)):
            X = gen_data.gen_data(n = 1000, tps = tp_comb, rhos = rhos[r])[0]
            R_org = latentcor.latentcor(X = X, tps = tp_comb, method = "original", use_nearPD = False)[1]
            R_approx = latentcor.latentcor(X = X, tps = tp_comb, method = "approx", use_nearPD = False)[1]
            rhorep[r] = rhos[r]; Rrep[r, 0] = R_org[1, 0]; Rrep[r, 1] = R_approx[1, 0]
        data = {"True latent correlation": rhorep, "Estimated latent correlation (original)": Rrep[ : , 0]}
        plot = seaborn.scatterplot(data = data, x = "True latent correlation", y = "Estimated latent correlation (original)")
        pyplot.plot(rhos, rhos, color = "r")
        pyplot.title(tp_comb[0] + " vs. " + tp_comb[1] + " (original)")
        pyplot.show()
        data = {"True latent correlation": rhorep, "Estimated latent correlation (approx)": Rrep[ : , 1]}
        plot = seaborn.scatterplot(data = data, x = "True latent correlation", y = "Estimated latent correlation (approx)")
        pyplot.plot(rhos, rhos, color = "r")
        pyplot.title(tp_comb[0] + " vs. " + tp_comb[1] + " (approx)")
        pyplot.show()"""
            

