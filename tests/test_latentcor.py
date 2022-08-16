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
    assert latentcor.gen_data(tps = ["con", "bin", "tru", "ter"], XP = None)['X'].shape[0] == 100
    assert latentcor.gen_data(n = 50, tps = ["con", "bin", "tru", "ter"], XP = None)['X'].shape[0] == 50
    assert latentcor.gen_data(tps = ["con", "bin", "tru", "ter"], XP = None)['X'].shape[1] == 4
    assert latentcor.gen_data(tps = ["bin", "bin"], XP = None)['X'].shape[1] == 2
    assert numpy.array_equiv(numpy.unique(latentcor.gen_data(tps = ["bin"])['X']), numpy.array([0, 1]))
    assert numpy.min(latentcor.gen_data(tps = ["tru"])['X']) == 0
    assert numpy.array_equiv(numpy.unique(latentcor.gen_data(tps = ["ter"])['X']), numpy.array([0, 1, 2]))
    
def test_get_tps():
    assert numpy.array_equiv(latentcor.get_tps(latentcor.gen_data()['X']), numpy.array(["ter", "con"]))
    assert numpy.array_equiv(latentcor.get_tps(latentcor.gen_data(tps = ["con", "con"])['X']), numpy.array(["con", "con"]))
    assert numpy.array_equiv(latentcor.get_tps(latentcor.gen_data(tps = ["con", "bin"])['X']), numpy.array(["con", "bin"]))
    assert numpy.array_equiv(latentcor.get_tps(latentcor.gen_data(tps = ["con", "tru"])['X']), numpy.array(["con", "tru"]))
    assert numpy.array_equiv(latentcor.get_tps(latentcor.gen_data(tps = ["con", "ter"])['X']), numpy.array(["con", "ter"]))
    assert numpy.array_equiv(latentcor.get_tps(latentcor.gen_data(tps = ["bin", "bin"])['X']), numpy.array(["bin", "bin"]))
    assert numpy.array_equiv(latentcor.get_tps(latentcor.gen_data(tps = ["bin", "tru"])['X']), numpy.array(["bin", "tru"]))
    assert numpy.array_equiv(latentcor.get_tps(latentcor.gen_data(tps = ["bin", "ter"])['X']), numpy.array(["bin", "ter"]))
    assert numpy.array_equiv(latentcor.get_tps(latentcor.gen_data(tps = ["tru", "tru"])['X']), numpy.array(["tru", "tru"]))
    assert numpy.array_equiv(latentcor.get_tps(latentcor.gen_data(tps = ["tru", "ter"])['X']), numpy.array(["tru", "ter"]))
    assert numpy.array_equiv(latentcor.get_tps(latentcor.gen_data(tps = ["ter", "ter"])['X']), numpy.array(["ter", "ter"]))
    assert numpy.array_equiv(latentcor.get_tps(latentcor.gen_data(tps = ["con", "bin", "tru", "ter"])['X']), numpy.array(["con", "bin", "tru", "ter"]))
def test_latentcor():
    X = latentcor.gen_data(tps = ["con", "con"])['X']
    assert numpy.array_equiv(latentcor.latentcor(X = X, tps = ["con", "con"], method = "original", nu = 0.5, tol = 1e-8, ratio = .9, showplot = False)['R'],
               latentcor.latentcor(X = X, tps = ["con", "con"], method = "original", nu = 0.5, tol = 1e-8, ratio = .9, showplot = False)['R'].T)
    assert numpy.array_equiv(latentcor.latentcor(X = X, tps = ["con", "con"], nu = 0.5, tol = 1e-8, ratio = .9, showplot = False)['R'],
               latentcor.latentcor(X = X, tps = ["con", "con"], nu = 0.5, tol = 1e-8, ratio = .9, showplot = False)['R'].T)
    X = latentcor.gen_data(tps = ["bin", "bin"])['X']
    assert numpy.array_equiv(latentcor.latentcor(X = X, tps = ["bin", "bin"], method = "original", nu = 0.5, tol = 1e-8, ratio = .9, showplot = False)['R'],
               latentcor.latentcor(X = X, tps = ["bin", "bin"], method = "original", nu = 0.5, tol = 1e-8, ratio = .9, showplot = False)['R'].T)
    assert numpy.array_equiv(latentcor.latentcor(X = X, tps = ["bin", "bin"], nu = 0.5, tol = 1e-8, ratio = .9, showplot = False)['R'],
               latentcor.latentcor(X = X, tps = ["bin", "bin"], nu = 0.5, tol = 1e-8, ratio = .9, showplot = False)['R'].T)
    X = latentcor.gen_data(tps = ["tru", "tru"])['X']
    assert numpy.array_equiv(latentcor.latentcor(X = X, tps = ["tru", "tru"], method = "original", nu = 0.5, tol = 1e-8, ratio = .9, showplot = False)['R'],
               latentcor.latentcor(X = X, tps = ["tru", "tru"], method = "original", nu = 0.5, tol = 1e-8, ratio = .9, showplot = False)['R'].T)
    assert numpy.array_equiv(latentcor.latentcor(X = X, tps = ["tru", "tru"], nu = 0.5, tol = 1e-8, ratio = .9, showplot = False)['R'],
               latentcor.latentcor(X = X, tps = ["tru", "tru"], nu = 0.5, tol = 1e-8, ratio = .9, showplot = False)['R'].T)
    X = latentcor.gen_data(tps = ["ter", "ter"])['X']
    assert numpy.array_equiv(latentcor.latentcor(X = X, tps = ["ter", "ter"], method = "original", nu = 0.5, tol = 1e-8, ratio = .9, showplot = False)['R'],
               latentcor.latentcor(X = X, tps = ["ter", "ter"], method = "original", nu = 0.5, tol = 1e-8, ratio = .9, showplot = False)['R'].T)
    assert numpy.array_equiv(latentcor.latentcor(X = X, tps = ["ter", "ter"], nu = 0.5, tol = 1e-8, ratio = .9, showplot = False)['R'],
               latentcor.latentcor(X = X, tps = ["ter", "ter"], nu = 0.5, tol = 1e-8, ratio = .9, showplot = False)['R'].T)
    X = latentcor.gen_data(tps = ["con", "bin"])['X']
    assert numpy.array_equiv(latentcor.latentcor(X = X, tps = ["con", "bin"], method = "original", nu = 0.5, tol = 1e-8, ratio = .9, showplot = False)['R'],
               latentcor.latentcor(X = X, tps = ["con", "bin"], method = "original", nu = 0.5, tol = 1e-8, ratio = .9, showplot = False)['R'].T)
    assert numpy.array_equiv(latentcor.latentcor(X = X, tps = ["con", "bin"], nu = 0.5, tol = 1e-8, ratio = .9, showplot = False)['R'],
               latentcor.latentcor(X = X, tps = ["con", "bin"], nu = 0.5, tol = 1e-8, ratio = .9, showplot = False)['R'].T)
    X = latentcor.gen_data(tps = ["tru", "bin"])['X']
    lat_tb= latentcor.latentcor(X = X, tps = ["tru", "bin"], method = "original", nu = 0.5, tol = 1e-8, ratio = .9, showplot = False)['R']
    assert numpy.array_equiv(lat_tb, lat_tb.T)
    assert numpy.array_equiv(latentcor.latentcor(X = X, tps = ["tru", "bin"], nu = 0.5, tol = 1e-8, ratio = .9, showplot = False)['R'],
               latentcor.latentcor(X = X, tps = ["tru", "bin"], nu = 0.5, tol = 1e-8, ratio = .9, showplot = False)['R'].T)
    X = latentcor.gen_data(tps = ["tru", "con"])['X']
    lat_tc = latentcor.latentcor(X = X, tps = ["tru", "con"], method = "original", nu = 0.5, tol = 1e-8, ratio = .9, showplot = False)['Rpointwise']
    assert numpy.array_equiv(lat_tc, lat_tc.T)
    assert numpy.array_equiv(numpy.around(latentcor.latentcor(X = X, tps = ["tru", "con"], nu = 0.5, tol = 1e-8, ratio = .9, showplot = False)['R'], 5),
               numpy.around(latentcor.latentcor(X = X, tps = ["tru", "con"], nu = 0.5, tol = 1e-8, ratio = .9, showplot = False)['R'].T, 5))
    X = latentcor.gen_data(tps = ["ter", "con"])['X']
    lat_nc = latentcor.latentcor(X = X, tps = ["ter", "con"], method = "original", nu = 0.5, tol = 1e-8, ratio = .9, showplot = False)['R']
    assert numpy.array_equiv(lat_nc, lat_nc.T)
    assert numpy.array_equiv(latentcor.latentcor(X = X, tps = ["ter", "con"], nu = 0.5, tol = 1e-8, ratio = .9, showplot = False)['R'],
               latentcor.latentcor(X = X, tps = ["ter", "con"], nu = 0.5, tol = 1e-8, ratio = .9, showplot = False)['R'].T)
    X = latentcor.gen_data(tps = ["ter", "bin"])['X']
    assert numpy.array_equiv(latentcor.latentcor(X = X, tps = ["ter", "bin"], method = "original", nu = 0.5, tol = 1e-8, ratio = .9, showplot = False)['R'],
               latentcor.latentcor(X = X, tps = ["ter", "bin"], method = "original", nu = 0.5, tol = 1e-8, ratio = .9, showplot = False)['R'].T)
    assert numpy.array_equiv(latentcor.latentcor(X = X, tps = ["ter", "bin"], nu = 0.5, tol = 1e-8, ratio = .9, showplot = False)['R'],
               latentcor.latentcor(X = X, tps = ["ter", "bin"], nu = 0.5, tol = 1e-8, ratio = .9, showplot = False)['R'].T)
    X = latentcor.gen_data(tps = ["ter", "tru"])['X']
    lat_nt = latentcor.latentcor(X = X, tps = ["ter", "tru"], method = "original", nu = 0.5, tol = 1e-8, ratio = .9, showplot = False)['R']
    assert numpy.array_equiv(lat_nt, lat_nt.T)
    assert numpy.array_equiv(latentcor.latentcor(X = X, tps = ["ter", "tru"], nu = 0.5, tol = 1e-8, ratio = .9, showplot = False)['R'],
               latentcor.latentcor(X = X, tps = ["ter", "tru"], nu = 0.5, tol = 1e-8, ratio = .9, showplot = False)['R'].T)
    X = latentcor.gen_data(tps = ["ter", "ter"])['X']
    assert numpy.array_equiv(latentcor.latentcor(X = X, tps = ["ter", "ter"], method = "original", nu = 0.5, tol = 1e-8, ratio = .9, showplot = False)['R'],
               latentcor.latentcor(X = X, tps = ["ter", "ter"], method = "original", nu = 0.5, tol = 1e-8, ratio = .9, showplot = False)['R'].T)
    assert numpy.array_equiv(latentcor.latentcor(X = X, tps = ["ter", "ter"], nu = 0.5, tol = 1e-8, ratio = .9, showplot = False)['R'],
               latentcor.latentcor(X = X, tps = ["ter", "ter"], nu = 0.5, tol = 1e-8, ratio = .9, showplot = False)['R'].T)
              

