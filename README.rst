
latentcor: Latent Correlation for Mixed Types of Data
=====================================================

.. image:: https://readthedocs.org/projects/latentcor-py/badge/?version=latest
        :target: https://latentcor-py.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

.. image:: https://img.shields.io/pypi/v/latentcor.svg
        :target: https://pypi.python.org/pypi/latentcor

.. image:: https://img.shields.io/travis/mingzehuang/latentcor.svg
        :target: https://travis-ci.com/mingzehuang/latentcor


Introduction
------------

`latentcor` is an Python package for estimation of latent correlations with mixed data types (continuous, binary, truncated, and ternary) under the latent Gaussian copula model. For references on the estimation framework, see

* `Fan, J., Liu, H., Ning, Y., and Zou, H. (2017), “High Dimensional Semiparametric Latent Graphical Model for Mixed Data.” <https://doi.org/10.1111/rssb.12168>`_ *JRSS B*. **Continuous/binary** types.

* `Quan X., Booth J.G. and Wells M.T. “Rank-based approach for estimating correlations in mixed ordinal data.” <https://arxiv.org/abs/1809.06255>`_ *arXiv*. **Ternary** type.

* `Yoon G., Carroll R.J. and Gaynanova I. (2020). “Sparse semiparametric canonical correlation analysis for data of mixed types.” <https://doi.org/10.1093/biomet/asaa007>`_ *Biometrika*. **Truncated** type for zero-inflated data.

* `Yoon G., Müller C.L. and Gaynanova I. (2021). “Fast computation of latent correlations.” <https://doi.org/10.1080/10618600.2021.1882468>`_. **Approximation method of computation**, see `math framework <https://mingzehuang.github.io/latentcor/articles/latentcor_math.html>`_ for details.


Statement of need
-----------------

No Python software package is currently available that allows accurate and fast correlation estimation from mixed variable data in a unifying manner. The Python package *latentcor*, introduced here, thus represents the first stand-alone R package for computation of latent correlation that takes into account all variable types (continuous/binary/ordinal/zero-inflated), comes with an optimized memory footprint, and is computationally efficient, essentially making latent correlation estimation almost as fast as rank-based correlation estimation.


* Free software: GNU General Public License v3
* Documentation: https://latentcor.readthedocs.io.


Installation
------------

The easiest way to install `latentcor` is using `pip`

.. code::

    pip install latentcor


Example
-------

First, we will generate a pair of variables with different types using a sample size :code:`n=100` which will serve as example data. Here first variable will be ternary, and second variable will be continuous::
    
    simdata = gen_data(n = 100, tps = ["ter", "con"])

The output of `gen_data` is a list with 2 elements:

* :code:`simdata[0]`: a matrix (:math:`100\times 2`), the first column is the ternary variable; the second column is the continuous variable.

* :code:`simdata[1]`: :code:`None` (:code:`showplot = False`, can be changed to display the plot of generated data in :code:`gen_data` input).

Then we can estimate the latent correlation matrix based on these 2 variables using :code:`latentcor` function::

    estimate = latentcor(simdata[0], tps = ["ter", "con"])

The output of :code:`estimate` is a list with several elements:

* :code:`estimate[0]`: estimated final latent correlation matrix, this matrix is guaranteed to be strictly positive definite (through :code:`tatsmodels.stats.correlation_tools.corr_nearest` projection and parameter :code:`nu`, see Mathematical framework for estimation) if :code:`use.nearPD = True`.

* :code:`estimate[1]`: matrix of pointwise estimated correlations. Due to pointwise estimation, it is not guaranteed to be positive semi-definite

* :code:`estimate[2]`: :code:`None` by default as :code:`showplot = False` in :code:`latentcor`. Otherwise displays a heatmap of latent correlation matrix.

* :code:`estimate[3]`: Kendall :math:`\tau (\tau_{a})` correlation matrix for these :math:`2` variables.

* :code:`estimate[4]`: a list has the same length as the number of variables. Here the first element is a (:math:`2\times1`) vector indicating the cumulative proportions for zeros and ones in the ternary variable (e.g. first element in vector is the proportion of zeros, second element in vector is the proportion of zeros and ones.) The second element of the list is :code:`numpy.na` for continuous variable.


Community Guidelines
--------------------

* Contributions and suggestions to the software are always welcome. Please consult our `contribution guidelines <https://github.com/mingzehuang/latentcor_py/blob/master/CONTRIBUTING.rst>`_ prior to submitting a pull request.
* Report issues or problems with the software using github’s `issue tracker <https://github.com/mingzehuang/latentcor_py/issues>`_.
* Contributors must adhere to the `Code of Conduct <https://github.com/mingzehuang/latentcor_py/blob/master/CODE_OF_CONDUCT.rst>`_.
* The easiest way to replicate development environment of `latentcor` is using `pip`:

.. code::

    pip install -r requirements_dev.txt


Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
