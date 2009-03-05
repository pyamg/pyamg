"""PyAMG: Algebraic Multigrid Solvers in Python"""


from info import __doc__

try:
    from __config__ import show as show_config
except ImportError, e:
    msg = """Error importing pyamg: you cannot import pyamg while
    being in pyamg source directory; please exit the pyamg source
    tree first, and relaunch your python intepreter."""
    raise ImportError(msg)


# Emit a warning if numpy is too old 
import numpy as _numpy
majver, minver = [float(i) for i in _numpy.version.version.split('.')[:2]] 
if majver < 1 or (majver == 1 and minver < 2): 
    import warnings 
    warnings.warn("Numpy 1.2.0 or above is recommended for this version of " \
                  "PyAMG (detected version %s)" % _numpy.version.version,
                  UserWarning) 

# Emit a warning if scipy is too old 
import scipy as _scipy
majver, minver = [float(i) for i in _scipy.version.version.split('.')[:2]] 
if minver < 0.7:
    import warnings 
    warnings.warn("SciPy 0.7 or above is recommended for this version of " \
                  "PyAMG (detected version %s)" % _scipy.version.version, 
                  UserWarning) 
del _numpy, _scipy


from version import version as __version__

from multilevel import *
from classical import ruge_stuben_solver
from aggregation import smoothed_aggregation_solver
from gallery import demo

__all__ = filter(lambda s:not s.startswith('_'),dir())
__all__ += ['test', '__version__']


from pyamg.testing import Tester
test = Tester().test
bench = Tester().bench

__doc__ += """

Utility tools
-------------

  test        --- Run pyamg unittests (requires nose unittest framework)
  bench       --- Run pyamg benchmarks (requires nose unittest framework) 
  __version__ --- pyamg version string

"""
