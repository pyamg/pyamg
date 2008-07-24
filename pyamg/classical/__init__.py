"""Classical AMG"""

#from info import __doc__

from classical import *
from split import *
from interpolate import *
from cr import *

__all__ = filter(lambda s:not s.startswith('_'),dir())
from pyamg.testing import Tester
test = Tester().test
