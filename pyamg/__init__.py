"Multigrid Solvers"

from info import __doc__

from multilevel import *
from rs import ruge_stuben_solver
from sa import smoothed_aggregation_solver
from gallery import *

from version import version as __version__

__all__ = filter(lambda s:not s.startswith('_'),dir())
from scipy.testing.pkgtester import Tester
test = Tester().test
