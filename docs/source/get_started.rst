Get started
===========

A simple example with two variables
-----------------------------------

Let's import :code:`gen_data`, :code:`get_tps` and :code:`latentcor` from package :code:`latentcor`.

.. jupyter-execute::

    from latentcor import gen_data, get_tps, latentcor

First, we will generate a pair of variables with different types using a sample size :code:`n=100` which will serve as example data. Here first variable will be ternary, and second variable will be continuous.

.. jupyter-execute::
    
    simdata = gen_data(n = 100, tps = ["ter", "con"])
    print(simdata['X'][ :6, :])

.. jupyter-execute::

    simdata['plotX']

The output of `gen_data` is a list with 2 elements:

* :code:`simdata['X']`: a matrix (:math:`100\times 2`), the first column is the ternary variable; the second column is the continuous variable.

* :code:`simdata['plotX']`: :code:`None` (:code:`showplot = False`, can be changed to display the plot of generated data in :code:`gen_data` input).

Then we use :code:`get_tps` to guess data types automatically.

.. jupyter-execute::

    data_types = get_tps(simdata['X'])
    print(data_types) 

Then we can estimate the latent correlation matrix based on these 2 variables using :code:`latentcor` function.

.. jupyter-execute::

    estimate = latentcor(simdata['X'], tps = data_types)
    print(estimate['R'])

.. jupyter-execute::

    print(estimate['Rpointwise'])

.. jupyter-execute::

    print(estimate['plot']) 

.. jupyter-execute::

    print(estimate['K'])

.. jupyter-execute::
    
    print(estimate['zratios'])

The output of :code:`estimate` is a list with several elements:

* :code:`estimate['R']`: estimated final latent correlation matrix, this matrix is guaranteed to be strictly positive definite (through :code:`statsmodels.stats.correlation_tools.corr_nearest` projection and parameter :code:`nu`, see Mathematical framework for estimation) if :code:`use.nearPD = True`.

* :code:`estimate['Rpointwise']`: matrix of pointwise estimated correlations. Due to pointwise estimation, it is not guaranteed to be positive semi-definite

* :code:`estimate['plot']`: :code:`None` by default as :code:`showplot = False` in :code:`latentcor`. Otherwise displays a heatmap of latent correlation matrix.

* :code:`estimate['K']`: Kendall :math:`\tau (\tau_{a})` correlation matrix for these :math:`2` variables.

* :code:`estimate['zratios']`: a list has the same length as the number of variables. Here the first element is a (:math:`2\times1`) vector indicating the cumulative proportions for zeros and ones in the ternary variable (e.g. first element in vector is the proportion of zeros, second element in vector is the proportion of zeros and ones.) The second element of the list is :code:`numpy.nan` for continuous variable.




   
