"""Classical AMG"""

from .classical import *
from .cr import *
from .interpolate import *
from .split import *

__all__ = [s for s in dir() if not s.startswith('_')]
