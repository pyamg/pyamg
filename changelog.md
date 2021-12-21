# Changelog

A summary of changes for pyamg.

## [unreleased]

## [4.2.0] - 2021-12-21

### Added
- add linting to ci
- add cibuildwheel to ci
- use pybind11 build extensions
- use scm for versioning
- add rr, rr+, MrMr, rMr convergence options to kyrlov solvers

### Changed
- consistent use of norms in Krylov
- update setuptools to setup.cfg

### Removed
- drop python 3.5
