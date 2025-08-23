# Changelog

A summary of changes for PyAMG.

## [5.3.0]
- add python 3.10-3.13
- handle new/old scipy return args in make_system
- add generalized Lloyd aggregation
- fix/replace scipy sparse calls
- fix docstrings throughout
- fix SOR bug
- drop pylint, use ruff for linting
- change names [bc]sr_matrix -> [bc]sr_array, etc.
- use @ for mat-vecs
- expand testing
- add pymetis to opt requirements; skip testing if unavailable
- add option to remove dirichlet portion
- add Floyd-Warshall and tests

## [5.2.1]
- update wheel building and versioning

## [5.2.0]
- use rtol, in Krylov solvers
- require scipy >= 1.11.0
- require python >= 3.9
- address numpy 2.0

## [5.1.0]
- use rtol, if it exists, in Krylov solvers
- spellcheck!
- add python 3.12 support
- require scipy >= 1.8.0
- require python >= 3.8
- address numpy/scipy deprecations

## [5.0.1]
- update JOSS article

## [5.0.0]
- add AIR solver
- add and extend classical AMG interpolation routines
- add support for F/C relaxation
- add block strength measures
- refactor smoothing interface
- add advection examples to the gallery
- add pairwise aggregation

## [4.2.3]
- add joss paper
- update dostrings throughout
- address minor bugs
- update readthedocs
- remove pytest as an install requires
- improve testing/features in fem.py

## [4.2.2]
- update import for scipy 1.8
- add python 3.10 to CI
- fix up docstrings
