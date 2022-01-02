#!/usr/bin/env python

"""Tests for `latentcor` package."""
import os
import sys
import pytest
sys.path.insert(0, os.path.abspath('../latentcor'))
from latentcor import latentcor, gen_data, get_tps


@pytest.fixture
def response():
    """Sample pytest fixture.

    See more at: http://doc.pytest.org/en/latest/fixture.html
    """
    # import requests
    # return requests.get('https://github.com/audreyr/cookiecutter-pypackage')

X = gen_data(n = 100, rhos = .5, copulas = ["no"], tps = ["con", "bin", "tru", "ter"], XP = None)[0]

def test_content(response):
    """Sample pytest test function with the pytest fixture as an argument."""
    
    assert X.shape[1] == 4
    # from bs4 import BeautifulSoup
    # assert 'GitHub' in BeautifulSoup(response.content).title.string

print(latentcor(X = X, tps = ["con", "bin", "tru", "ter"], method = "original", use_nearPD = False, nu = .1, tol = .001, ratio = .5)[0])
