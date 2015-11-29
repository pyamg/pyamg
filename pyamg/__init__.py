"""PyAMG: Algebraic Multigrid Solvers in Python"""
from __future__ import absolute_import

from .info import __doc__

__doc__ += """

Utility tools
-------------

  test        --- Run pyamg unittests (requires nose unittest framework)
  bench       --- Run pyamg benchmarks (requires nose unittest framework)
  __version__ --- pyamg version string

"""

# We first need to detect if we're being called as part of the pyamg setup
# procedure itself in a reliable manner.
try:
    __PYAMG_SETUP__
except NameError:
    __PYAMG_SETUP__ = False


if __PYAMG_SETUP__:
    import sys as _sys
    _sys.stderr.write('Running from pyamg source directory.\n')
    del _sys
else:
    try:
        from pyamg.__config__ import show as show_config
    except ImportError as e:
        msg = """Error importing pyamg: you cannot import pyamg while
        being in pyamg source directory; please exit the pyamg source
        tree first, and relaunch your python interpreter."""
        raise ImportError(msg)

    # Emit a warning if numpy is too old
    import numpy as np
    majver, minver = [float(i) for i in np.version.version.split('.')[:2]]
    if majver < 1 or (majver == 1 and minver < 2):
        import warnings
        warnings.warn("Numpy 1.2.0 or above is recommended for this version of\
                      PyAMG (detected version %s)" % np.version.version,
                      UserWarning)

    # Emit a warning if scipy is too old
    import scipy as sp
    majver, minver = [float(i) for i in sp.version.version.split('.')[:2]]
    if minver < 0.7:
        import warnings
        warnings.warn("SciPy 0.7 or above is recommended for this version of\
                      PyAMG (detected version %s)" % sp.version.version,
                      UserWarning)

    del np, sp

    from .version import git_revision as __git_revision__
    from .version import version as __version__

    from .multilevel import *
    from .classical import ruge_stuben_solver
    from .aggregation import smoothed_aggregation_solver, rootnode_solver
    from .gallery import demo
    from .blackbox import solve, solver, solver_configuration

    __all__ = [s for s in dir() if not s.startswith('_')]
    __all__ += ['test', '__version__']
    from numpy.testing import Tester
    test = Tester().test
    bench = Tester().bench
