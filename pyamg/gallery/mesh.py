"""Generates simple meshes"""

__docformat__ = "restructuredtext en"

import numpy

__all__ = ['regular_triangle_mesh']

def regular_triangle_mesh(nx, ny):
    """Construct a regular triangular mesh in the unit square
    
    Parameters
    ----------
    nx : int
       Number of nodes in the x direction
    ny : int
       Number of nodes in the y direction

    Returns
    -------
    Vert : array
        nx*ny x 2 vertex list
    E2V : array
        Nex x 3 element list

    Examples
    --------
    >>> E2V,Vert = regular_triangle_mesh(3, 2)

    """
    nx,ny = int(nx),int(ny)

    if nx < 2 or ny < 2:
        raise ValueError('minimum mesh dimension is 2: %s' % ((nx,ny),) )

    Vert1 = numpy.tile(numpy.arange(0, nx-1), ny - 1) + numpy.repeat(numpy.arange(0, nx * (ny - 1), nx), nx - 1)
    Vert3 = numpy.tile(numpy.arange(0, nx-1), ny - 1) + numpy.repeat(numpy.arange(0, nx * (ny - 1), nx), nx - 1) + nx
    Vert2 = Vert3 + 1
    Vert4 = Vert1 + 1

    Verttmp = numpy.meshgrid(numpy.arange(0, nx, dtype='float'), numpy.arange(0, ny, dtype='float'))
    Verttmp = (Verttmp[0].ravel(),Verttmp[1].ravel())
    Vert = numpy.vstack(Verttmp).transpose()
    Vert[:,0] = (1.0 / (nx - 1)) * Vert[:,0]
    Vert[:,1] = (1.0 / (ny - 1)) * Vert[:,1]

    E2V1 = numpy.vstack((Vert1,Vert2,Vert3)).transpose()
    E2V2 = numpy.vstack((Vert1,Vert4,Vert2)).transpose()
    E2V = numpy.vstack((E2V1,E2V2))

    return Vert,E2V
