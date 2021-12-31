Get started
===========

A simple example with two variables
-----------------------------------

First, we will generate a pair of variables with different types using a sample size :math:`n=100` which will serve as example data. Here first variable will be ternary, and second variable will be continuous::
    simdata = gen_data(n = 100, tps = ["ter", "con"])

The output of `gen_data` is a list with 2 elements::
    names(simdata)

`X`: a matrix (:math:`100\times 2`), the first column is the ternary variable; the second column is the continuous variable.
