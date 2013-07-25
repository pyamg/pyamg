"""Relaxation methods"""

from info import __doc__

from relaxation import *

__all__ = [s for s in dir() if not s.startswith('_')]
from pyamg.testing import Tester
test = Tester().test
