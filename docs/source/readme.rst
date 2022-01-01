README
======

Introduction
------------

No Python software package is currently available that allows accurate and fast correlation estimation from mixed variable data in a unifying manner. The Python package :code:`latentcor`, introduced here, thus represents the first stand-alone R package for computation of latent correlation that takes into account all variable types (continuous/binary/ordinal/zero-inflated), comes with an optimized memory footprint, and is computationally efficient, essentially making latent correlation estimation almost as fast as rank-based correlation estimation.
Python package :code:`latentcor` utilizes the powerful semi-parametric latent Gaussian copula models to estimate latent correlations between mixed data types :cite:p:`fan2017high,quan2018rank,yoon2020sparse,yoon2021fast`. The package allows to estimate correlations between any of continuous/binary/ternary/zero-inflated (truncated) variable types. The underlying implementation takes advantage of fast multi-linear interpolation scheme with a clever choice of grid points that give the package a small memory footprint, and allows to use the latent correlations with sub-sampling and bootstrapping.

* Free software: `GNU General Public License v3 <https://github.com/mingzehuang/latentcor_py/blob/master/LICENSE>`_
* Documentation: https://latentcor-py.readthedocs.io.

Installation
------------

The easiest way to install :code:`latentcor` is using :code:`pip`:

.. code::

    pip install latentcor

Statement of Need
-----------------

No Python software package is currently available that allows accurate and fast correlation estimation from mixed variable data in a unifying manner. The Python package :code:`latentcor`, introduced here, thus represents the first stand-alone Python package for computation of latent correlation that takes into account all variable types (continuous/binary/ordinal/zero-inflated), comes with an optimized memory footprint, 
and is computationally efficient, essentially making latent correlation estimation almost as fast as rank-based correlation estimation.

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

Reference
---------

.. bibliography::
