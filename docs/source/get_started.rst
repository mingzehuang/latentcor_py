Get started
===========

A simple example with two variables
-----------------------------------

Let's import :code:`gen_data`, :code:`get_tps` and :code:`latentcor` from package :code:`latentcor`.

.. code-block::

    >>> from latentcor import gen_data, get_tps, latentcor

First, we will generate a pair of variables with different types using a sample size :code:`n=100` which will serve as example data. Here first variable will be ternary, and second variable will be continuous.

.. code-block::
    
    >>> simdata = gen_data(n = 100, tps = ["ter", "con"])
    >>> print(simdata[0][ :6, :])
    [[ 2.          0.61200788]
     [ 0.         -2.00391226]
     [ 1.          0.26714693]
     [ 1.          0.17363031]
     [ 2.          2.70537882]
     [ 0.         -0.62657498]]
    >>> print(simdata[1])
    None

The output of `gen_data` is a list with 2 elements:

* :code:`simdata[0]`: a matrix (:math:`100\times 2`), the first column is the ternary variable; the second column is the continuous variable.

* :code:`simdata[1]`: :code:`None` (:code:`showplot = False`, can be changed to display the plot of generated data in :code:`gen_data` input).

Then we use :code:`get_tps` to guess data types automatically.

.. code-block::

    >>> data_types = get_tps(simdata[0])
    >>> print(data_types) 
    ['ter' 'con']

Then we can estimate the latent correlation matrix based on these 2 variables using :code:`latentcor` function.

.. code-block::

    >>> estimate = latentcor(simdata[0], tps = data_types)
    >>> print(estimate[0])
    [[0.001      0.56295584]
     [0.56295584 0.001     ]]
    >>> print(estimate[1])
    [[1.         0.56351936]
     [0.56351936 1.        ]]
    >>> print(estimate[2]) 
    None
    >>> print(estimate[3])
    [[1.        0.3139394]
     [0.3139394 1.       ]]
    >>> print(estimate[4])
    [[0.3 nan]
     [0.8 nan]]     

The output of :code:`estimate` is a list with several elements:

* :code:`estimate[0]`: estimated final latent correlation matrix, this matrix is guaranteed to be strictly positive definite (through :code:`statsmodels.stats.correlation_tools.corr_nearest` projection and parameter :code:`nu`, see Mathematical framework for estimation) if :code:`use.nearPD = True`.

* :code:`estimate[1]`: matrix of pointwise estimated correlations. Due to pointwise estimation, it is not guaranteed to be positive semi-definite

* :code:`estimate[2]`: :code:`None` by default as :code:`showplot = False` in :code:`latentcor`. Otherwise displays a heatmap of latent correlation matrix.

* :code:`estimate[3]`: Kendall :math:`\tau (\tau_{a})` correlation matrix for these :math:`2` variables.

* :code:`estimate[4]`: a list has the same length as the number of variables. Here the first element is a (:math:`2\times1`) vector indicating the cumulative proportions for zeros and ones in the ternary variable (e.g. first element in vector is the proportion of zeros, second element in vector is the proportion of zeros and ones.) The second element of the list is :code:`numpy.nan` for continuous variable.




   
