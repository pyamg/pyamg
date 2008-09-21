"""Relaxation methods"""

#TODO list relaxation methods and briefly explain role of relaxation in MG methods

#from info import __doc__

from relaxation import *

__all__ = filter(lambda s:not s.startswith('_'),dir())
from pyamg.testing import Tester
test = Tester().test
