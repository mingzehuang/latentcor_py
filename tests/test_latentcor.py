#!/usr/bin/env python

"""Tests for `latentcor` package."""

import pytest


from latentcor import latentcor
from latentcor import gen_data


@pytest.fixture
def response():
    """Sample pytest fixture.

    See more at: http://doc.pytest.org/en/latest/fixture.html
    """
    # import requests
    # return requests.get('https://github.com/audreyr/cookiecutter-pypackage')

X = gen_data.gen_data(n = 100, rhos = .5, copulas = ["no"], tps = ["con", "bin", "tru", "ter"], XP = None)[0]

def test_content(response):
    """Sample pytest test function with the pytest fixture as an argument."""
    
    assert X.shape[1] == 4
    # from bs4 import BeautifulSoup
    # assert 'GitHub' in BeautifulSoup(response.content).title.string

print(latentcor.latentcor(X = X, tps = ["con", "bin", "tru", "ter"], method = "original", use_nearPD = False, nu = .1, tol = .001, ratio = .5)[0])
