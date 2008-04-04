"""Classical AMG"""

#from info import __doc__

from classical import ruge_stuben_solver

__all__ = filter(lambda s:not s.startswith('_'),dir())
from scipy.testing.pkgtester import Tester
test = Tester().test
