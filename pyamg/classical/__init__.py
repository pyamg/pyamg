"""Classical AMG"""

from . import classical
from . import split
from . import interpolate
from . import cr

from .classical import ruge_stuben_solver

__all__ = [
    'classical',
    'split',
    'interpolate',
    'cr',
    'ruge_stuben_solver',
]
