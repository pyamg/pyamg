"""Utility functions."""

from . import linalg
from . import utils
from . import params
from . import mrhs

from .utils import make_system, upcast
from .mrhs import normalize_rhs, restore_shape, per_column_norms, all_converged

__all__ = ['all_converged', 'linalg', 'make_system', 'mrhs', 'normalize_rhs',
           'params', 'per_column_norms', 'restore_shape', 'upcast', 'utils']

__doc__ += """
linalg.py provides some linear algebra functionality not yet found in scipy.

utils.py provides some utility functions for use with pyamg

bsr_utils.py provides utility functions for accessing and writing individual
rows of BSR matrices

mrhs.py provides helpers for multiple right-hand side (MRHS) solves

"""
