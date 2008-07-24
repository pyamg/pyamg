"""Aggregation-based AMG"""

#from info import __doc__

from adaptive import *
from aggregate import *
from aggregation import *
from tentative import *
from smooth import *

__all__ = filter(lambda s:not s.startswith('_'),dir())
from pyamg.testing import Tester
test = Tester().test
