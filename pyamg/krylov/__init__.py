"Krylov methods for solving sparse linear systems"

from cg import *
from krylov import *

__all__ = filter(lambda s:not s.startswith('_'),dir())
from pyamg.testing import Tester
test = Tester().test
