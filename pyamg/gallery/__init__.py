"Matrix Gallery for Multigrid Solvers"

from info import __doc__

from elasticity import *
from example import *
from laplacian import *
from stencil import *
from sprand import *
from demo import demo

__all__ = filter(lambda s:not s.startswith('_'),dir())
from pyamg.testing import Tester
test = Tester().test
