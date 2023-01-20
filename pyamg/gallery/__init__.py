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

from .elasticity import linear_elasticity, linear_elasticity_p1
from .example import load_example
from .laplacian import poisson, gauge_laplacian
from .stencil import stencil_grid
from .mesh import regular_triangle_mesh
from .diffusion import diffusion_stencil_2d

from .random_sparse import sprand
from .demo import demo

__all__ = [
    'elasticity', 'laplacian', 'stencil', 'diffusion',
    'linear_elasticity', 'linear_elasticity_p1',
    'load_example',
    'poisson', 'gauge_laplacian',
    'stencil_grid',
    'regular_triangle_mesh',
    'diffusion_stencil_2d',
    'sprand',
    'demo',
]
