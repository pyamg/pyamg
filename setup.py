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
from pybind11.setup_helpers import Pybind11Extension, build_ext, has_flag

extra_compile_args = []
extra_link_args = []

for cflag, lflag in [('-fopenmp', '-fopenmp'),
                     ('-Xclang -fopenmp', '-lomp'),
                     ('-Xpreprocessor -fopenmp -lomp', '-lomp')] :
    try:
        openmp = has_flag(build_ext.compiler, cflag)
        extra_compile_args.append(cflag)
        extra_link_args.append(lflag)
    except:
        openmp = False

print(f'\n\nUsing openmp flag {extra_compile_args}\n\n')

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
