"Utility Functions"

from info import __doc__

from linalg import *
from utils import *

__all__ = filter(lambda s:not s.startswith('_'),dir())
from pyamg.testing import Tester
test = Tester().test
