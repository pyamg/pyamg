"""Classical AMG"""
from __future__ import absolute_import

from .classical import *
from .split import *
from .interpolate import *
from .cr import *
from numpy.testing import Tester

__all__ = [s for s in dir() if not s.startswith('_')]
test = Tester().test
