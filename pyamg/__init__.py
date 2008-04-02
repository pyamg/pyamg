"PyAMG: Algebraic Multigrid Solvers in Python"


from info import __doc__

try:
    from __config__ import show as show_config
except ImportError, e:
    msg = """Error importing pyamg: you cannot import pyamg while
    being in pyamg source directory; please exit the pyamg source
    tree first, and relaunch your python intepreter."""
    raise ImportError(msg)
from version import version as __version__


from multilevel import *
from classical import ruge_stuben_solver
from aggregation import smoothed_aggregation_solver
from gallery import *

__all__ = filter(lambda s:not s.startswith('_'),dir())
__all__ += ['test','__version__']


from scipy.testing.pkgtester import Tester
test = Tester().test
bench = Tester().bench
__doc__ += """

Utility tools
-------------

  test        --- Run pyamg unittests (requires nose unittest framework)
  __version__ --- pyamg version string

"""
