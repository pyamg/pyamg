"""Basic vtk support.

The vis module provides support for generic vtk file writing, basic mesh
writing (unstructured triangular and tetrahedral meshes), visualization of
aggregate groupings in 2d and in 3d, and C/F splittings.
"""

from . import vtk_writer
from . import vis_coarse

from .vtk_writer import write_vtu, write_basic_mesh
from .vis_coarse import vis_aggregate_groups

__all__ = [
    'vis_aggregate_groups',
    'vis_coarse',
    'vtk_writer',
    'write_basic_mesh',
    'write_vtu',
]
