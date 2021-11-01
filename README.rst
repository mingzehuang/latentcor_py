#############
latentcor: Latent Correlation for Mixed Types of Data
#############



.. image:: https://img.shields.io/pypi/v/latentcor.svg
        :target: https://pypi.python.org/pypi/latentcor

.. image:: https://img.shields.io/travis/mingzehuang/latentcor.svg
        :target: https://travis-ci.com/mingzehuang/latentcor

.. image:: https://readthedocs.org/projects/latentcor/badge/?version=latest
        :target: https://latentcor.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status

#. latentcor is an Python package for estimation of latent correlations with mixed data types (continuous, binary, truncated, and ternary) under the latent Gaussian copula model. For references on the estimation framework, see
     * [Fan, J., Liu, H., Ning, Y., and Zou, H. (2017), “High Dimensional Semiparametric Latent Graphical Model for Mixed Data.” *JRSS B*](https://doi.org/10.1111/rssb.12168). **Continuous/binary** types.
     * [Quan X., Booth J.G. and Wells M.T."Rank-based approach for estimating correlations in mixed ordinal data." *arXiv*](https://arxiv.org/abs/1809.06255) **Ternary** type.
     * [Yoon G., Carroll R.J. and Gaynanova I. (2020). “Sparse semiparametric canonical correlation analysis for data of mixed types”. *Biometrika*](https://doi.org/10.1093/biomet/asaa007). **Truncated** type for zero-inflated data.
     * [Yoon G., Müller C.L. and Gaynanova I. (2021). “Fast computation of latent correlations” *JCGS*](https://doi.org/10.1080/10618600.2021.1882468). **Approximation method of computation**, see [vignette](https://mingzehuang.github.io/latentcor/articles/latentcor.html) for details.

#. Statement of need
     * No R software package is currently available that allows accurate and fast correlation estimation from mixed variable data in a unifying manner. The R package [`latentcor`](https://CRAN.R-project.org/package=latentcor), introduced here, thus represents the first stand-alone R package for computation of latent correlation that takes into account all variable types (continuous/binary/ordinal/zero-inflated), comes with an optimized memory footprint, and is computationally efficient, essentially making latent correlation estimation almost as fast as rank-based correlation estimation.

     * Free software: GNU General Public License v3
     * Documentation: https://latentcor.readthedocs.io.

#. Installation


#. Example


Features
--------

* TODO

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
