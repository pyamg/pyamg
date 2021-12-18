"""PyAMG: Algebraic Multigrid Solvers in Python"""

import re
import warnings
import numpy as np
import scipy as sp

from .version import version_tuple as __version_tuple__
from .version import version as __version__

from . import (aggregation, amg_core, classical, gallery, krylov, relaxation, util, vis)
from . import (blackbox, graph, graph_ref, multilevel, strength)

from .multilevel import coarse_grid_solver, multilevel_solver, MultilevelSolver
from .classical import ruge_stuben_solver
from .aggregation import smoothed_aggregation_solver, rootnode_solver
from .gallery import demo
from .blackbox import solve, solver, solver_configuration

__all__ = ['__version_tuple__', '__version__',
           'aggregation', 'amg_core', 'classical', 'gallery', 'krylov', 'relaxation',
           'util', 'vis',
           'blackbox', 'graph', 'graph_ref', 'multilevel', 'strength',
           'coarse_grid_solver', 'multilevel_solver', 'MultilevelSolver',
           'ruge_stuben_solver', 'smoothed_aggregation_solver', 'rootnode_solver',
           'demo', 'solve', 'solver', 'solver_configuration']

__all__ += ['test']

__doc__ += """

Utility tools
-------------

  test        --- Run pyamg unittests (requires pytest)
  __version__ --- pyamg version string

"""

# Warn on old numpy or scipy.  Two digits.
npreq = '1.6'
npmin = [int(j) for j in npreq.split('.')]
m = re.match(r'(\d+)\.(\d+).*', np.__version__)
npver = [int(m.group(1)), int(m.group(2))]
if npver[0] < npmin[0] or (npver[0] == npmin[0] and npver[1] < npmin[1]):
    warnings.warn(f'Numpy {npmin} or above is recommended for this version of'
                  f'PyAMG (detected version {npver})', UserWarning)

spreq = '0.11'
spmin = [int(j) for j in spreq.split('.')]
m = re.match(r'(\d+)\.(\d+).*', sp.__version__)
spver = [int(m.group(1)), int(m.group(2))]
if spver[0] < spmin[0] or (spver[0] == spmin[0] and spver[1] < spmin[1]):
    warnings.warn(f'SciPy {spmin} or above is recommended for this version of'
                  f'PyAMG (detected version {spver})', UserWarning)


def test(verbose=False):
    """Test runner for pytest.

    Parameters
    ----------
    verbose : bool
        Turn on verbose output.

    """
    import sys     # pylint: disable=import-outside-toplevel
    import pytest  # pylint: disable=import-outside-toplevel

    sysversion = sys.version.replace('\n', '')
    print(f'Python version: {sysversion}')
    print(f'pytest version: {pytest.__version__}')
    print(f'scipy  version: {sp.__version__}')
    print(f'numpy  version: {np.__version__}')
    print(f'pyamg  version: {__version__}')

    pyamgdir = __path__[0]
    args = [pyamgdir]

    if verbose:
        args += ['--verbose']
    else:
        args += ['--quiet']

    try:
        return pytest.main(args)
    except SystemExit as e:
        return e.code
