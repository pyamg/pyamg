"""Aggregation-based AMG"""

from adaptive import *
from aggregate import *
from aggregation import *
from tentative import *
from smooth import *
from rootnode import *

__all__ = [s for s in dir() if not s.startswith('_')]
from numpy.testing import Tester
test = Tester().test
