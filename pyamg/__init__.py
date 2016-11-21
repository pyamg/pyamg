"""PyAMG: Algebraic Multigrid Solvers in Python"""
from __future__ import absolute_import

__doc__ += """

Utility tools
-------------

  test        --- Run pyamg unittests (requires nose unittest framework)
  bench       --- Run pyamg benchmarks (requires nose unittest framework)
  __version__ --- pyamg version string

"""

# Warn on old numpy or scipy
import numpy as np
npreq = '1.6'
npmin = [int(j) for j in npreq.split('.')]
npver = [int(j) for j in np.__version__.split('.')]
if npver[0] < npmin[0] or (npver[0] >= npmin[0] and npver[1] < npmin[1]):
    import warnings
    warnings.warn("Numpy %s or above is recommended for this version of"
                  "PyAMG (detected version %s)" % (npmin, npver),
                  UserWarning)

import scipy as sp
spreq = '0.11'
spmin = [int(j) for j in spreq.split('.')]
spver = [int(j) for j in sp.__version__.split('.')]
if spver[0] < spmin[0] or (spver[0] >= spmin[0] and spver[1] < spmin[1]):
    import warnings
    warnings.warn("SciPy %s or above is recommended for this version of"
                  "PyAMG (detected version %s)" % (spmin, spver),
                  UserWarning)

from .version import git_revision as __git_revision__
from .version import version as __version__

from .multilevel import *
from .classical import ruge_stuben_solver
from .aggregation import smoothed_aggregation_solver, rootnode_solver
from .gallery import demo
from .blackbox import solve, solver, solver_configuration

__all__ = [__git_revision__, __version__,
           ruge_stuben_solver, smoothed_aggregation_solver, rootnode_solver,
           demo, solve, solver, solver_configuration]

__all__ = [s for s in dir() if not s.startswith('_')]
__all__ += ['test', '__version__']

from numpy.testing import Tester
test = Tester().test
bench = Tester().bench
