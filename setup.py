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

# OpenMP support
from extension_helpers._openmp_helpers import get_openmp_flags, check_openmp_support

# try the automatic flags
extra_compile_args = []
extra_link_args = []
openmp_flags = get_openmp_flags()
openmpworks = check_openmp_support(openmp_flags=openmp_flags)
if openmpworks:
    extra_compile_args = openmp_flags.get('compiler_flags')
    extra_link_args = openmp_flags.get('linker_flags')

amg_core_headers = ['air',
                    'evolution_strength',
                    'graph',
                    'krylov',
                    'linalg',
                    'relaxation',
                    'ruge_stuben',
                    'smoothed_aggregation',
                    'sparse']

ext_modules = [
    Pybind11Extension(f'pyamg.amg_core.{f}',
                      extra_compile_args=extra_compile_args,
                      extra_link_args=extra_link_args,
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

if openmpworks:
    print('+++++++++++++++++\n   OpenMP enabled\n+++++++++++++++++')
else:
    print('-----------------\n   OpenMP not enabled\n-----------------')
print(extra_compile_args)
print(extra_link_args)
