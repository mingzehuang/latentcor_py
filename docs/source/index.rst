.. latentcor documentation master file, created by
   sphinx-quickstart on Fri Dec 31 00:21:43 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to latentcor's documentation!
=====================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

Introduction
============

R package `latentcor` utilizes the powerful semi-parametric latent Gaussian copula models to estimate latent correlations between mixed data types. The package allows to estimate correlations between any of continuous/binary/ternary/zero-inflated (truncated) variable types. The underlying implementation takes advantage of fast multi-linear interpolation scheme with a clever choice of grid points that give the package a small memory footprint, and allows to use the latent correlations with sub-sampling and bootstrapping.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
