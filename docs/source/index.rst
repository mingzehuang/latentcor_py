.. latentcor documentation master file, created by
   sphinx-quickstart on Fri Dec 31 00:21:43 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

latentcor: Latent Correlation for Mixed Types of Data
=====================================================

.. toctree::

   Introduction
   Statement of Need
   Get Started

Introduction
------------

R package `latentcor` utilizes the powerful semi-parametric latent Gaussian copula models to estimate latent correlations between mixed data types. The package allows to estimate correlations between any of continuous/binary/ternary/zero-inflated (truncated) variable types. The underlying implementation takes advantage of fast multi-linear interpolation scheme with a clever choice of grid points that give the package a small memory footprint, and allows to use the latent correlations with sub-sampling and bootstrapping.


Statement of Need
-----------------

No R software package is currently available that allows accurate and fast correlation estimation from mixed variable data in a unifying manner. The R package `latentcor`, introduced here, thus represents the first stand-alone R package for 
computation of latent correlation that takes into account all variable types (continuous/binary/ordinal/zero-inflated), comes with an optimized memory footprint, 
and is computationally efficient, essentially making latent correlation estimation almost as fast as rank-based correlation estimation. 


Getting Started
---------------







Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
