"""Generates simple meshes"""


import numpy as np

__all__ = ['regular_triangle_mesh']


def regular_triangle_mesh(nx, ny):
    """Construct a regular triangular mesh in the unit square

    Parameters
    ----------
    nx : int
       Number of nodes in the x-direction
    ny : int
       Number of nodes in the y-direction

    Returns
    -------
    Vert : array
        nx*ny x 2 vertex list
    E2V : array
        Nex x 3 element list

    Examples
    --------
    >>> from pyamg.gallery import regular_triangle_mesh
    >>> E2V,Vert = regular_triangle_mesh(3, 2)

    """
    nx, ny = int(nx), int(ny)

    if nx < 2 or ny < 2:
        raise ValueError('minimum mesh dimension is 2: %s' % ((nx, ny),))

    Vert1 = np.tile(np.arange(0, nx-1), ny - 1) +\
        np.repeat(np.arange(0, nx * (ny - 1), nx), nx - 1)
    Vert3 = np.tile(np.arange(0, nx-1), ny - 1) +\
        np.repeat(np.arange(0, nx * (ny - 1), nx), nx - 1) + nx
    Vert2 = Vert3 + 1
    Vert4 = Vert1 + 1

    Verttmp = np.meshgrid(np.arange(0, nx, dtype='float'),
                          np.arange(0, ny, dtype='float'))
    Verttmp = (Verttmp[0].ravel(), Verttmp[1].ravel())
    Vert = np.vstack(Verttmp).transpose()
    Vert[:, 0] = (1.0 / (nx - 1)) * Vert[:, 0]
    Vert[:, 1] = (1.0 / (ny - 1)) * Vert[:, 1]

    E2V1 = np.vstack((Vert1, Vert2, Vert3)).transpose()
    E2V2 = np.vstack((Vert1, Vert4, Vert2)).transpose()
    E2V = np.vstack((E2V1, E2V2))

    return Vert, E2V
