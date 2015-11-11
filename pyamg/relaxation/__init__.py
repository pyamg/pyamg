"""Relaxation methods"""
from __future__ import absolute_import

from .info import __doc__

__all__ = [s for s in dir() if not s.startswith('_')]
from numpy.testing import Tester
test = Tester().test
