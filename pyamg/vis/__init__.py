"Visualization Support"

from info import __doc__

from vtk_writer import *
from vis_coarse import *

__all__ = filter(lambda s:not s.startswith('_'),dir())
from pyamg.testing import Tester
test = Tester().test
