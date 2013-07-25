"""Classical AMG"""

from classical import *
from split import *
from interpolate import *
from cr import *

__all__ = [s for s in dir() if not s.startswith('_')]
from pyamg.testing import Tester
test = Tester().test
