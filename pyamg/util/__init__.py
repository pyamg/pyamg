"Utility Functions"

from info import __doc__

from linalg import *
from utils import *

__all__ = [s for s in dir() if not s.startswith('_')]
from numpy.testing import Tester
test = Tester().test
