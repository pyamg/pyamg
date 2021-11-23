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

from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext

install_requires = (
    'numpy>=1.7.0',
    'scipy>=0.12.0',
    'pytest>=2',
)

amg_core_headers = ['evolution_strength',
                    'graph',
                    'krylov',
                    'linalg',
                    'relaxation',
                    'ruge_stuben',
                    'smoothed_aggregation']

ext_modules = [
    Pybind11Extension(f'pyamg.amg_core.{f}',
                      sources=[f'pyamg/amg_core/{f}_bind.cpp'],
                     )
    for f in amg_core_headers]

ext_modules += [
    Pybind11Extension('pyamg.amg_core.tests.bind_examples',
                      sources=['pyamg/amg_core/tests/bind_examples_bind.cpp'],
                     )
    ]

setup(
    name='pyamg',
    use_scm_version={
        "version_scheme": lambda _: "4.1.0",
        "write_to": "pyamg/version.py"
    },
    keywords=['algebraic multigrid AMG sparse matrix preconditioning'],
    author='Nathan Bell, Luke Olson, and Jacob Schroder',
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
    package_data={'pyamg': ['gallery/example_data/*.mat', 'gallery/mesh_data/*.npz']},
    include_package_data=False,
    install_requires=install_requires,
    zip_safe=False,
    #
    ext_modules=ext_modules,
    cmdclass={'build_ext': build_ext},
    setup_requires=['numpy', 'pybind11', 'setuptools_scm'],
    #
    tests_require=['pytest'],
    #
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Environment :: Console',
        'Environment :: X11 Applications',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: C++',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Education',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
)
