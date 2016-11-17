#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from setuptools import setup, find_packages, Extension

name = 'pyamg'
author = 'Nathan Bell, Luke OLson, and Jacob Schroder'
author_email = 'luke.olson@gmail.com'
maintainer = 'Luke Olson'
maintainer_email = 'luke.olson@gmail.com'
url = 'https://github.com/pyamg/pyamg'
download_url = 'https://github.com/pyamg/pyamg/releases'

classifiers = """\
Development Status :: 5 - Production/Stable
Environment :: Console
Environment :: X11 Applications
Intended Audience :: Developers
Intended Audience :: Education
Intended Audience :: Science/Research
License :: OSI Approved :: MIT License
Natural Language :: English
Operating System :: OS Independent
Programming Language :: C++
Programming Language :: Python
Programming Language :: Python :: 2
Programming Language :: Python :: 2.6
Programming Language :: Python :: 2.7
Programming Language :: Python :: 3
Programming Language :: Python :: 3.3
Programming Language :: Python :: 3.4
Programming Language :: Python :: 3.5
Topic :: Education
Topic :: Scientific/Engineering
Topic :: Scientific/Engineering :: Mathematics
Topic :: Software Development :: Libraries :: Python Modules
"""

description = "PyAMG: Algebraic Multigrid Solvers in Python"
long_description = description + "\n\n" +\
    """
PyAMG is a library of Algebraic Multigrid (AMG)
solvers with a convenient Python interface.

PyAMG features implementations of:

- Ruge-Stuben (RS) or Classical AMG
- AMG based on Smoothed Aggregation (SA)
- Adaptive Smoothed Aggregation (αSA)
- Compatible Relaxation (CR)
- Krylov methods such as CG, GMRES, FGMRES, BiCGStab, MINRES, etc

PyAMG is primarily written in Python with
supporting C++ code for performance critical operations.
    """

classifiers = [_f for _f in classifiers.split('\n') if _f]
platforms = ['Windows', 'Linux', 'Solaris', 'Mac OS-X', 'Unix']
license = 'MIT'
version = '3.0.2'
install_requires = ['nose', 'numpy', 'scipy']
keywords = ['algebraic multigrid AMG sparse linear system preconditioning']
packages = find_packages(exclude=['doc'])
test_requirements = ['nose']

ext_modules = [Extension('pyamg.amg_core._amg_core',
                         sources=['pyamg/amg_core/amg_core_wrap.cxx'],
                         define_macros=[('__STDC_FORMAT_MACROS', 1)],
                         include_dirs=[np.get_include()])]

setup(
    name=name,
    version=version,
    description=description,
    long_description=long_description,
    author=author,
    author_email=author_email,
    url=url,
    #
    packages=packages,
    include_package_data=True,
    install_requires=install_requires,
    zip_safe=False,
    #
    license=license,
    keywords=keywords,
    classifiers=classifiers,
    #
    maintainer=maintainer,
    maintainer_email=maintainer_email,
    download_url=download_url,
    platforms=platforms,
    #
    test_suite='tests',
    test_requirements=test_requirements,
    #
    ext_modules=ext_modules,
    )
