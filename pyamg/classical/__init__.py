"""Classical AMG"""

from classical import *
from split import *
from interpolate import *
from cr import *

__all__ = [s for s in dir() if not s.startswith('_')]
from numpy.testing import Tester
test = Tester().test
