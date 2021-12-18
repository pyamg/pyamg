"""Utility Functions"""

from . import linalg
from . import utils
from . import params

__all__ = ['linalg', 'utils', 'params']

__doc__ += """
linalg.py provides some linear algebra functionality not yet found in scipy.

utils.py provides some utility functions for use with pyamg

bsr_utils.py provides utility functions for accessing and writing individual
rows of BSR matrices

"""
