"""
Fast Computation of Latent Correlations for Mixed Data
======================================================


The first stand-alone R package for computation of latent correlation that takes into account all variable types (continuous/binary/ordinal/zero-inflated), comes with an optimized memory footprint, and is computationally efficient, essentially making latent correlation estimation almost as fast as rank-based correlation estimation.
The estimation is based on latent copula Gaussian models.
For continuous/binary types, see Fan, J., Liu, H., Ning, Y., and Zou, H. (2017) <doi:10.1111/rssb.12168>.
For ternary type, see Quan X., Booth J.G. and Wells M.T. (2018) <arXiv:1809.06255>.
For truncated type or zero-inflated type, see Yoon G., Carroll R.J. and Gaynanova I. (2020) <doi:10.1093/biomet/asaa007>.
For approximation method of computation, see Yoon G., Müller C.L. and Gaynanova I. (2021) <doi:10.1080/10618600.2021.1882468>.
"""
import os
import sys
sys.path.insert(0, os.path.abspath('../latentcor'))

import os
import pickle
import lzma

"""ipol_10_file = pkg_resources.resource_stream('data', 'ipol_10.xz')"""
with lzma.open(os.path.join(os.path.abspath('../latentcor'), "data", "ipol_10.xz"), "rb") as f:
    ipol_10 = pickle.load(f)

"""ipol_11_file = pkg_resources.resource_stream('data', 'ipol_11.xz')"""
with lzma.open(os.path.join(os.path.abspath('../latentcor'), "data", "ipol_11.xz"), "rb") as f:
    ipol_11 = pickle.load(f)

"""ipol_20_file = pkg_resources.resource_stream('data', 'ipol_20.xz')"""
with lzma.open(os.path.join(os.path.abspath('../latentcor'), "data", "ipol_20.xz"), "rb") as f:
    ipol_20 = pickle.load(f)

"""ipol_21_file = pkg_resources.resource_stream('data', 'ipol_21.xz')"""
with lzma.open(os.path.join(os.path.abspath('../latentcor'), "data", "ipol_21.xz"), "rb") as f:
    ipol_21 = pickle.load(f)

"""ipol_22_file = pkg_resources.resource_stream('data', 'ipol_22.xz')"""
with lzma.open(os.path.join(os.path.abspath('../latentcor'), "data", "ipol_22.xz"), "rb") as f:
    ipol_22 = pickle.load(f)

"""ipol_30_file = pkg_resources.resource_stream('data', 'ipol_30.xz')"""
with lzma.open(os.path.join(os.path.abspath('../latentcor'), "data", "ipol_30.xz"), "rb") as f:
    ipol_30 = pickle.load(f)

"""ipol_31_file = pkg_resources.resource_stream('data', 'ipol_31.xz')"""
with lzma.open(os.path.join(os.path.abspath('../latentcor'), "data", "ipol_31.xz"), "rb") as f:
    ipol_31 = pickle.load(f)

"""ipol_32_file = pkg_resources.resource_stream('data', 'ipol_32.xz')"""
with lzma.open(os.path.join(os.path.abspath('../latentcor'), "data", "ipol_32.xz"), "rb") as f:
    ipol_32 = pickle.load(f)

"""ipol_33_file = pkg_resources.resource_stream('data', 'ipol_33.xz')"""
with lzma.open(os.path.join(os.path.abspath('../latentcor'), "data", "ipol_33.xz"), "rb") as f:
    ipol_33 = pickle.load(f)

"""
with lzma.open(os.path.join(os.getcwd(), "latentcor", "data", "all_ipol.xz"), "rb") as f:
    ipol_10, ipol_11, ipol_20, ipol_21, ipol_22, ipol_30, ipol_31, ipol_32, ipol_33 = pickle.load(f)
"""

__author__ = 'Mingze Huang, Christian L. Müller, Irina Gaynanova'
__email__ = 'mingzehuang@gmail.com, christian.mueller@stat.uni-muenchen.de, irinag@stat.tamu.edu'
__version__ = '0.1.0'

print(__version__)
