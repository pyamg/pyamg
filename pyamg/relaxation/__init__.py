"""Relaxation methods"""
from __future__ import absolute_import

from .info import __doc__

from numpy.testing import Tester

__all__ = [s for s in dir() if not s.startswith('_')]
test = Tester().test
