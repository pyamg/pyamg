"""Utility functions."""

from . import linalg
from . import utils
from . import params

try:
    # scipy >=1.8
    from scipy.sparse.linalg._isolve.utils import make_system
except ImportError:
    # scipy <1.8
    from scipy.sparse.linalg.isolve.utils import make_system

__all__ = ['linalg', 'utils', 'params', 'make_system']

__doc__ += """
linalg.py provides some linear algebra functionality not yet found in scipy.

utils.py provides some utility functions for use with pyamg

bsr_utils.py provides utility functions for accessing and writing individual
rows of BSR matrices

"""
