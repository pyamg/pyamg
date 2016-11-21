"""Aggregation-based AMG"""
from __future__ import absolute_import

from .adaptive import *
from .aggregate import *
from .aggregation import *
from .tentative import *
from .smooth import *
from .rootnode import *
from numpy.testing import Tester

__all__ = [s for s in dir() if not s.startswith('_')]
test = Tester().test
