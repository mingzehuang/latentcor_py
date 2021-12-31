Get started
===========

A simple example with two variables
-----------------------------------

First, we will generate a pair of variables with different types using a sample size :code:`n=100` which will serve as example data. Here first variable will be ternary, and second variable will be continuous::
    
    simdata = gen_data(n = 100, tps = ["ter", "con"])

The output of `gen_data` is a list with 2 elements:

* :code:`simdata[0]`: a matrix (:math:`100\times 2`), the first column is the ternary variable; the second column is the continuous variable.

* :code:`simdata[1]`: NULL (:code:`showplot = FALSE`, can be changed to display the plot of generated data in`gen_data` input).


   
