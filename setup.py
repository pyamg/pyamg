#!/usr/bin/env python
"""PyAMG: Algebraic Multigrid Solvers in Python.

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

from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

amg_core_headers = ['air',
                    'evolution_strength',
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
    ext_modules=ext_modules,
    cmdclass={'build_ext': build_ext},
)
