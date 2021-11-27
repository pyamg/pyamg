"Matrix Gallery for Multigrid Solvers"

from .demo import demo
from .diffusion import *
from .elasticity import *
from .example import *
from .info import __doc__
from .laplacian import *
from .mesh import *
from .random_sparse import *
from .stencil import *

__all__ = [s for s in dir() if not s.startswith('_')]
