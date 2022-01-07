#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst', encoding="utf8") as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst', encoding="utf8") as history_file:
    history = history_file.read()

requirements = ['numpy>=1.21', 'scipy>=1.7', 'statsmodels>=0.13', 'seaborn>=0.11', 'matplotlib>=3.5', 'joblib>=1.1']


setup(
    author="Mingze Huang, Christian L. Müller, Irina Gaynanova",
    author_email='mingzehuang@gmail.com, christian.mueller@stat.uni-muenchen.de, irinag@stat.tamu.edu',
    python_requires='>=3.7',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10'
    ],
    description="Fast Computation of Latent Correlations for Mixed Data",
    install_requires=requirements,
    license="GNU General Public License v3",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    package_data={'': ['*.xz']},
    keywords='latentcor',
    name='latentcor',
    packages=find_packages(include=['latentcor', 'latentcor.*']),
    test_suite='tests',
    url='https://github.com/mingzehuang/latentcor_py',
    version='0.1.3',
    zip_safe=False,
)
