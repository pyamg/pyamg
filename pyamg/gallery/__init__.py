"Matrix Gallery for Multigrid Solvers"

from info import __doc__

from elasticity import *
from example import *
from laplacian import *
from stencil import *
from mesh import *
from diffusion import *
from random_sparse import *
from demo import demo

__all__ = [s for s in dir() if not s.startswith('_')]
from numpy.testing import Tester
test = Tester().test
