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

from . import gen_data
from . import get_tps
from . import latentcor

__author__ = 'Mingze Huang, Christian L. Müller, Irina Gaynanova'
__email__ = 'mingzehuang@gmail.com, christian.mueller@stat.uni-muenchen.de, irinag@stat.tamu.edu'
__version__ = '1.0.0'
