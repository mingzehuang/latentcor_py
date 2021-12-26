#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst', encoding="utf8") as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst', encoding="utf8") as history_file:
    history = history_file.read()

requirements = ['numpy', 'scipy', 'statsmodels', 'seaborn', 'matplotlib', 'os', 'lzma', 'pickle', 'joblib']

setup_requirements = ['pytest-runner', ]

test_requirements = ['pytest>=3', ]

setup(
    author="Mingze Huang, Christian L. Müller, Irina Gaynanova",
    author_email='mingzehuang@gmail.com, christian.mueller@stat.uni-muenchen.de, irinag@stat.tamu.edu',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
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
    package_data={'': ['data/*']},
    keywords='latentcor',
    name='latentcor',
    packages=find_packages(include=['latentcor', 'latentcor.*']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/mingzehuang/latentcor_py',
    version='1.0.0',
    zip_safe=False,
)
