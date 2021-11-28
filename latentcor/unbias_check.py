import internal
import latentcor
import get_tps
import gen_data
import numpy
import seaborn
from matplotlib import pyplot
from scipy import stats


"""Ternary vs. Continuous"""
rhos = numpy.linspace(-1,1,100); rhorep = numpy.repeat(numpy.nan, len(rhos))
Rrep = numpy.repeat(numpy.nan, len(rhos) * 3).reshape(len(rhos), 3)
for r in range(len(rhos)):
    X = gen_data.gen_data(n = 100, tps = ["ter", "con"], rhos = rhos[r], XP = numpy.array([[.3, numpy.nan], [.5, numpy.nan]], dtype = float, ndmin = 2), showplot = False)[0]
    R_nc_org = latentcor.latentcor(X = X, tps = ["ter", "con"], method = "original", use_nearPD = False)[1]
    R_nc_approx = latentcor.latentcor(X = X, tps = ["ter", "con"], method = "approx", use_nearPD = False)[1]
    rhorep[r] = rhos[r]; Rrep[r, 0] = R_nc_org[1, 0]; Rrep[r, 1] = R_nc_approx[1, 0]; Rrep[r, 2] = numpy.corrcoef(X)[0,1]
print(Rrep)
print(rhorep)
data = {"True latent correlation": rhorep, "Estimated latent correlation (original)": Rrep[ : , 0]}
plot = seaborn.scatterplot(data = data, x = "True latent correlation", y = "Estimated latent correlation (original)")
pyplot.plot(rhos, rhos, color = "r")
pyplot.title("Ternary vs. Continuous (original)")
pyplot.show()

data = {"True latent correlation": rhorep, "Estimated latent correlation (approx)": Rrep[ : , 1]}
plot = seaborn.scatterplot(data = data, x = "True latent correlation", y = "Estimated latent correlation (approx)")
pyplot.plot(rhos, rhos, color = "r")
pyplot.title("Ternary vs. Continuous (approx)")
pyplot.show()


"""Binary vs. Continuous"""
rhos = numpy.linspace(-1,1,100); rhorep = numpy.repeat(numpy.nan, len(rhos))
Rrep = numpy.repeat(numpy.nan, len(rhos) * 3).reshape(len(rhos), 3)
for r in range(len(rhos)):
    X = gen_data.gen_data(n = 100, tps = ["bin", "con"], rhos = rhos[r], XP = numpy.array([[.5, numpy.nan], [numpy.nan, numpy.nan]], dtype = float, ndmin = 2), showplot = False)[0]
    R_nc_org = latentcor.latentcor(X = X, tps = ["bin", "con"], method = "original", use_nearPD = False)[1]
    R_nc_approx = latentcor.latentcor(X = X, tps = ["bin", "con"], method = "approx", use_nearPD = False)[1]
    rhorep[r] = rhos[r]; Rrep[r, 0] = R_nc_org[1, 0]; Rrep[r, 1] = R_nc_approx[1, 0]; Rrep[r, 2] = numpy.corrcoef(X)[0,1]
print(Rrep)
print(rhorep)
data = {"True latent correlation": rhorep, "Estimated latent correlation (original)": Rrep[ : , 0]}
plot = seaborn.scatterplot(data = data, x = "True latent correlation", y = "Estimated latent correlation (original)")
pyplot.plot(rhos, rhos, color = "r")
pyplot.title("Binary vs. Continuous (original)")
pyplot.show()

data = {"True latent correlation": rhorep, "Estimated latent correlation (approx)": Rrep[ : , 1]}
plot = seaborn.scatterplot(data = data, x = "True latent correlation", y = "Estimated latent correlation (approx)")
pyplot.plot(rhos, rhos, color = "r")
pyplot.title("Binary vs. Continuous (approx)")
pyplot.show()

"""Truncated vs. Continuous"""
rhos = numpy.linspace(-1,1,100); rhorep = numpy.repeat(numpy.nan, len(rhos))
Rrep = numpy.repeat(numpy.nan, len(rhos) * 3).reshape(len(rhos), 3)
for r in range(len(rhos)):
    X = gen_data.gen_data(n = 100, tps = ["tru", "con"], rhos = rhos[r], XP = numpy.array([[.5, numpy.nan], [numpy.nan, numpy.nan]], dtype = float, ndmin = 2), showplot = False)[0]
    R_nc_org = latentcor.latentcor(X = X, tps = ["tru", "con"], method = "original", use_nearPD = False)[1]
    R_nc_approx = latentcor.latentcor(X = X, tps = ["tru", "con"], method = "approx", use_nearPD = False)[1]
    rhorep[r] = rhos[r]; Rrep[r, 0] = R_nc_org[1, 0]; Rrep[r, 1] = R_nc_approx[1, 0]; Rrep[r, 2] = numpy.corrcoef(X)[0,1]
print(Rrep)
print(rhorep)
data = {"True latent correlation": rhorep, "Estimated latent correlation (original)": Rrep[ : , 0]}
plot = seaborn.scatterplot(data = data, x = "True latent correlation", y = "Estimated latent correlation (original)")
pyplot.plot(rhos, rhos, color = "r")
pyplot.title("Truncated vs. Continuous (original)")
pyplot.show()

data = {"True latent correlation": rhorep, "Estimated latent correlation (approx)": Rrep[ : , 1]}
plot = seaborn.scatterplot(data = data, x = "True latent correlation", y = "Estimated latent correlation (approx)")
pyplot.plot(rhos, rhos, color = "r")
pyplot.title("Truncated vs. Continuous (approx)")
pyplot.show()

"""Continuous vs. Continuous"""
rhos = numpy.linspace(-1,1,100); rhorep = numpy.repeat(numpy.nan, len(rhos))
Rrep = numpy.repeat(numpy.nan, len(rhos) * 3).reshape(len(rhos), 3)
for r in range(len(rhos)):
    X = gen_data.gen_data(n = 100, tps = ["con", "con"], rhos = rhos[r], XP = numpy.array([[.5, numpy.nan], [numpy.nan, numpy.nan]], dtype = float, ndmin = 2), showplot = False)[0]
    R_nc_org = latentcor.latentcor(X = X, tps = ["con", "con"], method = "original", use_nearPD = False)[1]
    R_nc_approx = latentcor.latentcor(X = X, tps = ["con", "con"], method = "approx", use_nearPD = False)[1]
    rhorep[r] = rhos[r]; Rrep[r, 0] = R_nc_org[1, 0]; Rrep[r, 1] = R_nc_approx[1, 0]; Rrep[r, 2] = numpy.corrcoef(X)[0,1]
print(Rrep)
print(rhorep)
data = {"True latent correlation": rhorep, "Estimated latent correlation (original)": Rrep[ : , 0]}
plot = seaborn.scatterplot(data = data, x = "True latent correlation", y = "Estimated latent correlation (original)")
pyplot.plot(rhos, rhos, color = "r")
pyplot.title("Continuous vs. Continuous (original)")
pyplot.show()

data = {"True latent correlation": rhorep, "Estimated latent correlation (approx)": Rrep[ : , 1]}
plot = seaborn.scatterplot(data = data, x = "True latent correlation", y = "Estimated latent correlation (approx)")
pyplot.plot(rhos, rhos, color = "r")
pyplot.title("Continuous vs. Continuous (approx)")
pyplot.show()

"""Need more check"""
"""Ternary vs. Binary"""
rhos = numpy.linspace(-1,1,100); rhorep = numpy.repeat(numpy.nan, len(rhos))
Rrep = numpy.repeat(numpy.nan, len(rhos) * 3).reshape(len(rhos), 3)
for r in range(len(rhos)):
    X = gen_data.gen_data(n = 100, tps = ["ter", "bin"], rhos = rhos[r], XP = numpy.array([[.3, .5], [.5, numpy.nan]], dtype = float, ndmin = 2), showplot = False)[0]
    R_nc_org = latentcor.latentcor(X = X, tps = ["ter", "bin"], method = "original", use_nearPD = False)[1]
    R_nc_approx = latentcor.latentcor(X = X, tps = ["ter", "bin"], method = "approx", use_nearPD = False)[1]
    rhorep[r] = rhos[r]; Rrep[r, 0] = R_nc_org[1, 0]; Rrep[r, 1] = R_nc_approx[1, 0]; Rrep[r, 2] = numpy.corrcoef(X)[0,1]
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
pyplot.show()

"""Binary vs. Binary"""
rhos = numpy.linspace(-1,1,100); rhorep = numpy.repeat(numpy.nan, len(rhos))
Rrep = numpy.repeat(numpy.nan, len(rhos) * 3).reshape(len(rhos), 3)
for r in range(len(rhos)):
    X = gen_data.gen_data(n = 100, tps = ["bin", "bin"], rhos = rhos[r], XP = numpy.array([[.5, .5], [numpy.nan, numpy.nan]], dtype = float, ndmin = 2), showplot = False)[0]
    R_nc_org = latentcor.latentcor(X = X, tps = ["bin", "bin"], method = "original", use_nearPD = False)[1]
    R_nc_approx = latentcor.latentcor(X = X, tps = ["bin", "bin"], method = "approx", use_nearPD = False)[1]
    rhorep[r] = rhos[r]; Rrep[r, 0] = R_nc_org[1, 0]; Rrep[r, 1] = R_nc_approx[1, 0]; Rrep[r, 2] = numpy.corrcoef(X)[0,1]
print(Rrep)
print(rhorep)
data = {"True latent correlation": rhorep, "Estimated latent correlation (original)": Rrep[ : , 0]}
plot = seaborn.scatterplot(data = data, x = "True latent correlation", y = "Estimated latent correlation (original)")
pyplot.plot(rhos, rhos, color = "r")
pyplot.title("Binary vs. Binary (original)")
pyplot.show()

data = {"True latent correlation": rhorep, "Estimated latent correlation (approx)": Rrep[ : , 1]}
plot = seaborn.scatterplot(data = data, x = "True latent correlation", y = "Estimated latent correlation (approx)")
pyplot.plot(rhos, rhos, color = "r")
pyplot.title("Binary vs. Binary (approx)")
pyplot.show()

"""Truncated vs. Binary"""
rhos = numpy.linspace(-1,1,100); rhorep = numpy.repeat(numpy.nan, len(rhos))
Rrep = numpy.repeat(numpy.nan, len(rhos) * 3).reshape(len(rhos), 3)
for r in range(len(rhos)):
    X = gen_data.gen_data(n = 100, tps = ["tru", "bin"], rhos = rhos[r], XP = numpy.array([[.5, .5], [numpy.nan, numpy.nan]], dtype = float, ndmin = 2), showplot = False)[0]
    R_nc_org = latentcor.latentcor(X = X, tps = ["tru", "bin"], method = "original", use_nearPD = False)[1]
    R_nc_approx = latentcor.latentcor(X = X, tps = ["tru", "bin"], method = "approx", use_nearPD = False)[1]
    rhorep[r] = rhos[r]; Rrep[r, 0] = R_nc_org[1, 0]; Rrep[r, 1] = R_nc_approx[1, 0]; Rrep[r, 2] = numpy.corrcoef(X)[0,1]
print(Rrep)
print(rhorep)
data = {"True latent correlation": rhorep, "Estimated latent correlation (original)": Rrep[ : , 0]}
plot = seaborn.scatterplot(data = data, x = "True latent correlation", y = "Estimated latent correlation (original)")
pyplot.plot(rhos, rhos, color = "r")
pyplot.title("Truncated vs. Binary (original)")
pyplot.show()

data = {"True latent correlation": rhorep, "Estimated latent correlation (approx)": Rrep[ : , 1]}
plot = seaborn.scatterplot(data = data, x = "True latent correlation", y = "Estimated latent correlation (approx)")
pyplot.plot(rhos, rhos, color = "r")
pyplot.title("Truncated vs. Binary (approx)")
pyplot.show()

"""Continuous vs. Binary"""
rhos = numpy.linspace(-1,1,100); rhorep = numpy.repeat(numpy.nan, len(rhos))
Rrep = numpy.repeat(numpy.nan, len(rhos) * 3).reshape(len(rhos), 3)
for r in range(len(rhos)):
    X = gen_data.gen_data(n = 100, tps = ["con", "bin"], rhos = rhos[r], XP = numpy.array([[numpy.nan, .5], [numpy.nan, numpy.nan]], dtype = float, ndmin = 2), showplot = False)[0]
    R_nc_org = latentcor.latentcor(X = X, tps = ["con", "bin"], method = "original", use_nearPD = False)[1]
    R_nc_approx = latentcor.latentcor(X = X, tps = ["con", "bin"], method = "approx", use_nearPD = False)[1]
    rhorep[r] = rhos[r]; Rrep[r, 0] = R_nc_org[1, 0]; Rrep[r, 1] = R_nc_approx[1, 0]; Rrep[r, 2] = numpy.corrcoef(X)[0,1]
print(Rrep)
print(rhorep)
data = {"True latent correlation": rhorep, "Estimated latent correlation (original)": Rrep[ : , 0]}
plot = seaborn.scatterplot(data = data, x = "True latent correlation", y = "Estimated latent correlation (original)")
pyplot.plot(rhos, rhos, color = "r")
pyplot.title("Continuous vs. Binary (original)")
pyplot.show()

data = {"True latent correlation": rhorep, "Estimated latent correlation (approx)": Rrep[ : , 1]}
plot = seaborn.scatterplot(data = data, x = "True latent correlation", y = "Estimated latent correlation (approx)")
pyplot.plot(rhos, rhos, color = "r")
pyplot.title("Continuous vs. Binary (approx)")
pyplot.show()


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

"""Binary vs. Truncated"""
rhos = numpy.linspace(-1,1,100); rhorep = numpy.repeat(numpy.nan, len(rhos))
Rrep = numpy.repeat(numpy.nan, len(rhos) * 3).reshape(len(rhos), 3)
for r in range(len(rhos)):
    X = gen_data.gen_data(n = 100, tps = ["bin", "tru"], rhos = rhos[r], XP = numpy.array([[.5, .5], [numpy.nan, numpy.nan]], dtype = float, ndmin = 2), showplot = False)[0]
    R_nc_org = latentcor.latentcor(X = X, tps = ["bin", "tru"], method = "original", use_nearPD = False)[1]
    R_nc_approx = latentcor.latentcor(X = X, tps = ["bin", "tru"], method = "approx", use_nearPD = False)[1]
    rhorep[r] = rhos[r]; Rrep[r, 0] = R_nc_org[1, 0]; Rrep[r, 1] = R_nc_approx[1, 0]; Rrep[r, 2] = numpy.corrcoef(X)[0,1]
print(Rrep)
print(rhorep)
data = {"True latent correlation": rhorep, "Estimated latent correlation (original)": Rrep[ : , 0]}
plot = seaborn.scatterplot(data = data, x = "True latent correlation", y = "Estimated latent correlation (original)")
pyplot.plot(rhos, rhos, color = "r")
pyplot.title("Binary vs. Truncated (original)")
pyplot.show()

data = {"True latent correlation": rhorep, "Estimated latent correlation (approx)": Rrep[ : , 1]}
plot = seaborn.scatterplot(data = data, x = "True latent correlation", y = "Estimated latent correlation (approx)")
pyplot.plot(rhos, rhos, color = "r")
pyplot.title("Binary vs. Truncated (approx)")
pyplot.show()

"""Truncated vs. Truncated"""
rhos = numpy.linspace(-1,1,100); rhorep = numpy.repeat(numpy.nan, len(rhos))
Rrep = numpy.repeat(numpy.nan, len(rhos) * 3).reshape(len(rhos), 3)
for r in range(len(rhos)):
    X = gen_data.gen_data(n = 100, tps = ["tru", "tru"], rhos = rhos[r], XP = numpy.array([[.5, .5], [numpy.nan, numpy.nan]], dtype = float, ndmin = 2), showplot = False)[0]
    R_nc_org = latentcor.latentcor(X = X, tps = ["tru", "tru"], method = "original", use_nearPD = False)[1]
    R_nc_approx = latentcor.latentcor(X = X, tps = ["tru", "tru"], method = "approx", use_nearPD = False)[1]
    rhorep[r] = rhos[r]; Rrep[r, 0] = R_nc_org[1, 0]; Rrep[r, 1] = R_nc_approx[1, 0]; Rrep[r, 2] = numpy.corrcoef(X)[0,1]
print(Rrep)
print(rhorep)
data = {"True latent correlation": rhorep, "Estimated latent correlation (original)": Rrep[ : , 0]}
plot = seaborn.scatterplot(data = data, x = "True latent correlation", y = "Estimated latent correlation (original)")
pyplot.plot(rhos, rhos, color = "r")
pyplot.title("Truncated vs. Truncated (original)")
pyplot.show()
data = {"True latent correlation": rhorep, "Estimated latent correlation (approx)": Rrep[ : , 1]}
plot = seaborn.scatterplot(data = data, x = "True latent correlation", y = "Estimated latent correlation (approx)")
pyplot.plot(rhos, rhos, color = "r")
pyplot.title("Truncated vs. Truncated (approx)")
pyplot.show()

"""Continuous vs. Truncated"""
rhos = numpy.linspace(-1,1,100); rhorep = numpy.repeat(numpy.nan, len(rhos))
Rrep = numpy.repeat(numpy.nan, len(rhos) * 3).reshape(len(rhos), 3)
for r in range(len(rhos)):
    X = gen_data.gen_data(n = 100, tps = ["con", "tru"], rhos = rhos[r], XP = numpy.array([[numpy.nan, .5], [numpy.nan, numpy.nan]], dtype = float, ndmin = 2), showplot = False)[0]
    R_nc_org = latentcor.latentcor(X = X, tps = ["con", "tru"], method = "original", use_nearPD = False)[1]
    R_nc_approx = latentcor.latentcor(X = X, tps = ["con", "tru"], method = "approx", use_nearPD = False)[1]
    rhorep[r] = rhos[r]; Rrep[r, 0] = R_nc_org[1, 0]; Rrep[r, 1] = R_nc_approx[1, 0]; Rrep[r, 2] = numpy.corrcoef(X)[0,1]
print(Rrep)
print(rhorep)
data = {"True latent correlation": rhorep, "Estimated latent correlation (original)": Rrep[ : , 0]}
plot = seaborn.scatterplot(data = data, x = "True latent correlation", y = "Estimated latent correlation (original)")
pyplot.plot(rhos, rhos, color = "r")
pyplot.title("Continuous vs. Truncated (original)")
pyplot.show()
data = {"True latent correlation": rhorep, "Estimated latent correlation (approx)": Rrep[ : , 1]}
plot = seaborn.scatterplot(data = data, x = "True latent correlation", y = "Estimated latent correlation (approx)")
pyplot.plot(rhos, rhos, color = "r")
pyplot.title("Continuous vs. Truncated (approx)")
pyplot.show()

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
rhos = numpy.linspace(-1,1,100); rhorep = numpy.repeat(numpy.nan, len(rhos))
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
pyplot.show()

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
rhos = numpy.linspace(-1,1,100); rhorep = numpy.repeat(numpy.nan, len(rhos))
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
pyplot.show()


