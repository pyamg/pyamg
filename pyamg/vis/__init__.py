"Visualization Support"

from info import __doc__

from vtk_writer import *
from vis_coarse import *

__all__ = [s for s in dir() if not s.startswith('_')]
from numpy.testing import Tester
test = Tester().test
