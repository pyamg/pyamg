"""Classical AMG."""

from . import air
from . import classical
from . import split
from . import interpolate
from . import cr

from .classical import ruge_stuben_solver
from .air import air_solver

__all__ = [
    'air',
    'air_solver',
    'classical',
    'cr',
    'interpolate',
    'ruge_stuben_solver',
    'split',
]
