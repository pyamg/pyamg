"Visualization Support"

from . import vtk_writer
from . import vis_coarse

__all__ = ['vtk_writer', 'vis_coarse']

__doc__ += """Basic vtk support.

The vis module provides support for generic vtk file writing, basic mesh
writing (unstructured triangular and tetrahedral meshes), visualization of
aggregate groupings in 2d and in 3d, and C/F splittings.
"""
