"""PyAMG: Algebraic Multigrid Solvers in Python."""

from . import (aggregation, amg_core, classical, gallery, krylov, relaxation, util, vis)
from . import (blackbox, graph, graph_ref, multilevel, strength)

from .multilevel import coarse_grid_solver, multilevel_solver, MultilevelSolver
from .classical import ruge_stuben_solver, air_solver
from .aggregation import smoothed_aggregation_solver, rootnode_solver, pairwise_solver
from .gallery import demo
from .blackbox import solve, solver, solver_configuration
from .util.sparse_blas import set_num_threads, get_num_threads, has_openmp

import importlib.metadata
__version__ = importlib.metadata.version(__name__)

__all__ = [
    'MultilevelSolver',
    '__version__',
    'aggregation',
    'air_solver',
    'amg_core',
    'blackbox',
    'classical',
    'coarse_grid_solver',
    'demo',
    'gallery',
    'get_num_threads',
    'graph',
    'graph_ref',
    'has_openmp',
    'krylov',
    'multilevel',
    'multilevel_solver',
    'pairwise_solver',
    'relaxation',
    'rootnode_solver',
    'ruge_stuben_solver',
    'set_num_threads',
    'smoothed_aggregation_solver',
    'solve',
    'solver',
    'solver_configuration',
    'strength',
    'util',
    'vis',
]

__all__ += ['test']

__doc__ += """

Utility tools
-------------
test         Run pyamg unittests (requires pytest)
__version__  pyamg version string
"""

from pyamg._tools._tester import PytestTester  # noqa
test = PytestTester(__name__)
del PytestTester
