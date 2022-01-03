#!/usr/bin/env python

"""Tests for `latentcor` package."""
import os
import sys
sys.path.insert(0, os.path.abspath('../latentcor'))
import pytest
import numpy
from latentcor import latentcor, gen_data, get_tps


@pytest.fixture
def response():
    """Sample pytest fixture.

    See more at: http://doc.pytest.org/en/latest/fixture.html
    """
    # import requests
    # return requests.get('https://github.com/audreyr/cookiecutter-pypackage')
def test_gen_data():
    assert gen_data(tps = ["con", "bin", "tru", "ter"], XP = None)[0].shape[0] == 100
    assert gen_data(n = 50, tps = ["con", "bin", "tru", "ter"], XP = None)[0].shape[0] == 50
    assert gen_data(tps = ["con", "bin", "tru", "ter"], XP = None)[0].shape[1] == 4
    assert gen_data(tps = ["bin", "bin"], XP = None)[0].shape[1] == 2
    assert numpy.array_equiv(numpy.unique(gen_data(tps = ["bin"])[0]), numpy.array([0, 1]))
    assert numpy.min(gen_data(tps = ["tru"])[0]) == 0
    assert numpy.array_equiv(numpy.unique(gen_data(tps = ["ter"])[0]), numpy.array([0, 1, 2]))
