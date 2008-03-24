"Matrix Gallery for Multigrid Solvers"

from info import __doc__

from elasticity import *
from example import *
from laplacian import *
from stencil import *

__all__ = filter(lambda s:not s.startswith('_'),dir())
from scipy.testing.pkgtester import Tester
test = Tester().test
