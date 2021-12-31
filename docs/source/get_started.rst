Get started
===========

A simple example with two variables
-----------------------------------

First, we will generate a pair of variables with different types using a sample size `n=100` which will serve as example data. Here first variable will be ternary, and second variable will be continuous.

data_generation::

    simdata = gen_data(n = 100, tps = ["ter", "con"])