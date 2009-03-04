"Krylov Solvers"


from info import __doc__

from _gmres import *
from _fgmres import *
from _cg import *
from _cgnr import *
from _cgne import *
from _bicgstab import *

__all__ = filter(lambda s:not s.startswith('_'),dir())
from pyamg.testing import Tester
test = Tester().test
