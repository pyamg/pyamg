"""Classical AMG."""

from . import classical
from . import air
from . import split
from . import interpolate
from . import cr
>>>>>>> 125040aff36fd22fc6ab523ca64d9954b1eb19fd

from .classical import ruge_stuben_solver
from .classical import air_solver

__all__ = [
    'classical',
    'air',
    'split',
    'interpolate',
    'cr',
    'ruge_stuben_solver',
    'air_solver',
]
