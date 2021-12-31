Get started
===========

A simple example with two variables
-----------------------------------

First, we will generate a pair of variables with different types using a sample size :code:`n=100` which will serve as example data. Here first variable will be ternary, and second variable will be continuous::
    
    simdata = gen_data(n = 100, tps = ["ter", "con"])

The output of `gen_data` is a list with 2 elements:

* :code:`simdata[0]`: a matrix (:math:`100\times 2`), the first column is the ternary variable; the second column is the continuous variable.

* :code:`simdata[1]`: :code:`None` (:code:`showplot = False`, can be changed to display the plot of generated data in :code:`gen_data` input).

Then we can estimate the latent correlation matrix based on these 2 variables using :code:`latentcor` function::

    estimate = latentcor(simdata[0], tps = ["ter", "con"])

The output of :code:`latentcor` is a list with several elements:

* :code:`latentcor[0]`: estimated final latent correlation matrix, this matrix is guaranteed to be strictly positive definite (through :code:`tatsmodels.stats.correlation_tools.corr_nearest` projection and parameter :code:`nu`, see Mathematical framework for estimation) if :code:`use.nearPD = True`.

* :code:`latentcor[1]`: matrix of pointwise estimated correlations. Due to pointwise estimation, it is not guaranteed to be positive semi-definite

* :code:`latentcor[2]`: :code:`None` by default as :code:`showplot = False` in :code:`latentcor`. Otherwise displays a heatmap of latent correlation matrix.

* :code:`latentcor[3]`: Kendall :math:`\tau (\tau_{a})` correlation matrix for these :math:`2` variables.

* :code:`latentcor[4]`: a list has the same length as the number of variables. Here the first element is a (:math:`2\times1`) vector indicating the cumulative proportions for zeros and ones in the ternary variable (e.g. first element in vector is the proportion of zeros, second element in vector is the proportion of zeros and ones.) The second element of the list is :code:`numpy.na` for continuous variable.




   
