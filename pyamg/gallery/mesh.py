"""Construct sparse matrix from a local stencil"""

__docformat__ = "restructuredtext en"

from numpy import tile, arange, repeat, hstack, meshgrid,vstack

__all__ = ['uniform_tri']

def uniform_tri(nx,ny):
    """Construct a regular triangular grid in the unit square
    
    Parameters
    ----------
    nx,ny : int
       number of nodes in each direction 

    Returns
    -------
    Vert : array
        nx*ny x 2 vertex list
    E2V : array
        Nex x 3 element list

    Examples
    --------
    >>> E2V,Vert = get_reg_tri(5,5)
    """
    Vert1 = tile(arange(0,nx-1),ny-1) + repeat(arange(0,nx*(ny-1),nx),nx-1)
    Vert3 = tile(arange(0,nx-1),ny-1) + repeat(arange(0,nx*(ny-1),nx),nx-1) + nx
    Vert2 = Vert3 + 1
    Vert4 = Vert1 + 1
    Verttmp = meshgrid(arange(0,nx,dtype='float'),arange(0,ny,dtype='float'))
    Verttmp = (Verttmp[0].ravel(),Verttmp[1].ravel())
    Vert = vstack(Verttmp).transpose()
    Vert[:,0] = (1.0/(nx-1))*Vert[:,0]
    Vert[:,1] = (1.0/(ny-1))*Vert[:,1]
    E2V1 = vstack((Vert1,Vert2,Vert3)).transpose()
    E2V2 = vstack((Vert1,Vert4,Vert2)).transpose()
    E2V = vstack((E2V1,E2V2))
    return Vert,E2V
