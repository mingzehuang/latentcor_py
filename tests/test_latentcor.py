#!/usr/bin/env python

"""Tests for `latentcor` package."""
import os
import sys
sys.path.insert(0, os.path.abspath('.'))
import pytest
import numpy
import latentcor


@pytest.fixture
def response():
    """Sample pytest fixture.

    See more at: http://doc.pytest.org/en/latest/fixture.html
    """
    # import requests
    # return requests.get('https://github.com/audreyr/cookiecutter-pypackage')
def test_gen_data():
    assert latentcor.gen_data(tps = ["con", "bin", "tru", "ter"], XP = None)[0].shape[0] == 100
    assert latentcor.gen_data(n = 50, tps = ["con", "bin", "tru", "ter"], XP = None)[0].shape[0] == 50
    assert latentcor.gen_data(tps = ["con", "bin", "tru", "ter"], XP = None)[0].shape[1] == 4
    assert latentcor.gen_data(tps = ["bin", "bin"], XP = None)[0].shape[1] == 2
    assert numpy.array_equiv(numpy.unique(latentcor.gen_data(tps = ["bin"])[0]), numpy.array([0, 1]))
    assert numpy.min(latentcor.gen_data(tps = ["tru"])[0]) == 0
    assert numpy.array_equiv(numpy.unique(latentcor.gen_data(tps = ["ter"])[0]), numpy.array([0, 1, 2]))
    
def test_get_tps():
    assert numpy.array_equiv(latentcor.get_tps(latentcor.gen_data()[0]), numpy.array(["ter", "con"]))

def test_latentcor():
    X = latentcor.gen_data(tps = ["con", "con"])[0]
    assert numpy.array_equiv(latentcor.latentcor(X = X, tps = ["con", "con"], method = "original", nu = 0.5, tol = 1e-8, ratio = .9, showplot = False)[0],
               latentcor.latentcor(X = X, tps = ["con", "con"], method = "original", nu = 0.5, tol = 1e-8, ratio = .9, showplot = False)[0].T)
    X = latentcor.gen_data(tps = ["bin", "bin"])[0]
    assert numpy.array_equiv(latentcor.latentcor(X = X, tps = ["bin", "bin"], method = "original", nu = 0.5, tol = 1e-8, ratio = .9, showplot = False)[0],
               latentcor.latentcor(X = X, tps = ["bin", "bin"], method = "original", nu = 0.5, tol = 1e-8, ratio = .9, showplot = False)[0].T)
    X = latentcor.gen_data(tps = ["bin", "bin"])[0]
    assert numpy.array_equiv(latentcor.latentcor(X = X, tps = ["bin", "bin"], nu = 0.5, tol = 1e-8, ratio = .9, showplot = False)[0],
               latentcor.latentcor(X = X, tps = ["bin", "bin"], nu = 0.5, tol = 1e-8, ratio = .9, showplot = False)[0].T)
    X = latentcor.gen_data(tps = ["tru", "tru"])[0]
    assert numpy.array_equiv(latentcor.latentcor(X = X, tps = ["tru", "tru"], method = "original", nu = 0.5, tol = 1e-8, ratio = .9, showplot = False)[0],
               latentcor.latentcor(X = X, tps = ["tru", "tru"], method = "original", nu = 0.5, tol = 1e-8, ratio = .9, showplot = False)[0].T)
    X = latentcor.gen_data(tps = ["tru", "tru"])[0]
    assert numpy.array_equiv(latentcor.latentcor(X = X, tps = ["tru", "tru"], nu = 0.5, tol = 1e-8, ratio = .9, showplot = False)[0],
               latentcor.latentcor(X = X, tps = ["tru", "tru"], nu = 0.5, tol = 1e-8, ratio = .9, showplot = False)[0].T)
    X = latentcor.gen_data(tps = ["ter", "ter"])[0]
    assert numpy.array_equiv(latentcor.latentcor(X = X, tps = ["ter", "ter"], method = "original", nu = 0.5, tol = 1e-8, ratio = .9, showplot = False)[0],
               latentcor.latentcor(X = X, tps = ["ter", "ter"], method = "original", nu = 0.5, tol = 1e-8, ratio = .9, showplot = False)[0].T)
    X = latentcor.gen_data(tps = ["ter", "ter"])[0]
    assert numpy.array_equiv(latentcor.latentcor(X = X, tps = ["ter", "ter"], nu = 0.5, tol = 1e-8, ratio = .9, showplot = False)[0],
               latentcor.latentcor(X = X, tps = ["ter", "ter"], nu = 0.5, tol = 1e-8, ratio = .9, showplot = False)[0].T)
