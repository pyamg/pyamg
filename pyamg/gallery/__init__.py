"""Matrix gallery of model problems.

Functions
---------
    - poisson() : Poisson problem using Finite Differences
    - linear_elasticity() : Linear Elasticity using Finite Elements
    - stencil_grid() : General stencil generation from 1D, 2D, and 3D
    - diffusion_stencil_2d() : 2D rotated anisotropic FE/FD stencil
"""

from . import elasticity
from . import laplacian
from . import stencil
from . import diffusion
from . import fem

from .elasticity import linear_elasticity, linear_elasticity_p1
from .example import load_example
from .laplacian import poisson, gauge_laplacian
from .stencil import stencil_grid
from .mesh import regular_triangle_mesh
from .diffusion import diffusion_stencil_2d
from .advection import advection_2d

from .random_sparse import sprand  # note: could use scipy.sparse.random_array
from .demo import demo

__all__ = [
    'advection_2d',
    'demo',
    'diffusion',
    'diffusion_stencil_2d',
    'elasticity',
    'fem',
    'gauge_laplacian',
    'laplacian',
    'linear_elasticity',
    'linear_elasticity_p1',
    'load_example',
    'poisson',
    'regular_triangle_mesh',
    'sprand',
    'stencil',
    'stencil_grid',
]
