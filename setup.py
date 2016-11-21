#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PyAMG: Algebraic Multigrid Solvers in Python

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

import io
import version_tools
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext as _build_ext

version = '3.1.1'
isreleased = False

# set the version information
fullversion, git_revision, git_branch =\
    version_tools.set_version_info(version, isreleased)
version_tools.write_version_py(version, fullversion,
                               git_revision, git_branch,
                               isreleased,
                               filename='pyamg/version.py')


# identify extension modules
# since numpy is needed (for the path), need to bootstrap the setup
# http://stackoverflow.com/questions/19919905/how-to-bootstrap-numpy-installation-in-setup-py
class build_ext(_build_ext):
    'to install numpy'
    def finalize_options(self):
        _build_ext.finalize_options(self)
        __builtins__.__NUMPY_SETUP__ = False
        import numpy
        self.include_dirs.append(numpy.get_include())

ext_modules = [Extension('pyamg.amg_core._amg_core',
                         sources=['pyamg/amg_core/amg_core_wrap.cxx'],
                         define_macros=[('__STDC_FORMAT_MACROS', 1)])]

setup(
    name='pyamg',
    version=fullversion,
    keywords=['algebraic multigrid AMG sparse matrix preconditioning'],
    author='Nathan Bell, Luke OLson, and Jacob Schroder',
    author_email='luke.olson@gmail.com',
    maintainer='Luke Olson',
    maintainer_email='luke.olson@gmail.com',
    url='https://github.com/pyamg/pyamg',
    download_url='https://github.com/pyamg/pyamg/releases',
    license='MIT',
    platforms=['Windows', 'Linux', 'Solaris', 'Mac OS-X', 'Unix'],
    description=__doc__.split('\n')[0],
    long_description=__doc__,
    #
    packages=find_packages(exclude=['doc']),
    package_data={'pyamg': ['gallery/example_data/*.mat']},
    include_package_data=False,
    install_requires=io.open('requirements.txt').read().split('\n'),
    zip_safe=False,
    #
    ext_modules=ext_modules,
    cmdclass={'build_ext': build_ext},
    setup_requires=['numpy'],
    #
    test_suite='tests',
    tests_require=['nose'],
    #
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Environment :: Console",
        "Environment :: X11 Applications",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: C++",
        "Programming Language :: Python",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 2.6",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Topic :: Education",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
