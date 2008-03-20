"PyAMG: Algebraic Multigrid Solvers in Python"

__all__ = ['test','__version__']

from info import __doc__

try:
    from __config__ import show as show_config
except ImportError, e:
    msg = """Error importing pyamg: you cannot import scipy while
    being in pyamg source directory; please exit the pyamg source
    tree first, and relaunch your python intepreter."""
    raise ImportError(msg)
from version import version as __version__


from multilevel import *
from rs import ruge_stuben_solver
from sa import smoothed_aggregation_solver
from gallery import *


from scipy.testing.pkgtester import Tester
test = Tester().test
bench = Tester().bench
__doc__ += """

Utility tools
-------------

  test        --- Run pyamg unittests (requires nose unittest framework)
  __version__ --- pyamg version string

"""
