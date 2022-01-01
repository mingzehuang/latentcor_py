#############
latentcor: Latent Correlation for Mixed Types of Data
#############

.. image:: https://readthedocs.org/projects/latentcor-py/badge/?version=latest
        :target: https://latentcor-py.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

.. image:: https://img.shields.io/pypi/v/latentcor.svg
        :target: https://pypi.python.org/pypi/latentcor

.. image:: https://img.shields.io/travis/mingzehuang/latentcor.svg
        :target: https://travis-ci.com/mingzehuang/latentcor


*************
Introduction
*************

``latentcor`` is an Python package for estimation of latent correlations with mixed data types (continuous, binary, truncated, and ternary) under the latent Gaussian copula model. For references on the estimation framework, see

* `Fan, J., Liu, H., Ning, Y., and Zou, H. (2017), “High Dimensional Semiparametric Latent Graphical Model for Mixed Data.” <https://doi.org/10.1111/rssb.12168>`_ *JRSS B*. **Continuous/binary** types.

* `Quan X., Booth J.G. and Wells M.T. “Rank-based approach for estimating correlations in mixed ordinal data.” <https://arxiv.org/abs/1809.06255>`_ *arXiv*. **Ternary** type.

* `Yoon G., Carroll R.J. and Gaynanova I. (2020). “Sparse semiparametric canonical correlation analysis for data of mixed types.” <https://doi.org/10.1093/biomet/asaa007>`_ *Biometrika*. **Truncated** type for zero-inflated data.

* `Yoon G., Müller C.L. and Gaynanova I. (2021). “Fast computation of latent correlations.” <https://doi.org/10.1080/10618600.2021.1882468>`_. **Approximation method of computation**, see `math framework <https://mingzehuang.github.io/latentcor/articles/latentcor_math.html>`_ for details.

*************
Statement of need
*************

No Python software package is currently available that allows accurate and fast correlation estimation from mixed variable data in a unifying manner. The Python package *latentcor*, introduced here, thus represents the first stand-alone R package for computation of latent correlation that takes into account all variable types (continuous/binary/ordinal/zero-inflated), comes with an optimized memory footprint, and is computationally efficient, essentially making latent correlation estimation almost as fast as rank-based correlation estimation.


#. Free software: GNU General Public License v3
#. Documentation: https://latentcor.readthedocs.io.

*************
Installation
*************
..
  The easiest way to install ``latentcor`` is using ``pip``   ::

    pip install latentcor


*************
Example
*************

*************
Reproducibility (for Developer)
*************
..
  The easiest way to replicate development environment of ``latentcor`` is using ``pip``   ::

    pip install -r requirements_dev.txt


*************
Community Guidelines
*************

* Contributions and suggestions to the software are always welcome. Please consult our `contribution guidelines <https://github.com/mingzehuang/latentcor_py/blob/master/CONTRIBUTING.rst>`_ prior to submitting a pull request.
* Report issues or problems with the software using github’s `issue tracker <https://github.com/mingzehuang/latentcor_py/issues>`_.
* Contributors must adhere to the `Code of Conduct <https://github.com/mingzehuang/latentcor_py/blob/master/CODE_OF_CONDUCT.rst>`_.



Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
