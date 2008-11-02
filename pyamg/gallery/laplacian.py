"""Discretizations of the Poisson problem"""

__docformat__ = "restructuredtext en"

__all__ = ['poisson']

import numpy as np

from stencil import stencil_grid

def poisson( grid, spacing=None, dtype=float, format=None):
    """Returns a sparse matrix for the N-dimensional Poisson problem

    The matrix represents a finite Difference approximation to the 
    Poisson problem on a regular n-dimensional grid with unit grid 
    spacing and Dirichlet boundary conditions.

    Parameters
    ----------
    grid : tuple of integers
        grid dimensions e.g. (100,100)

    Notes
    -----
    The matrix is symmetric and positive definite (SPD).

    Examples
    --------
    >>> # 4 nodes in one dimension
    >>> poisson( (4,) ).todense()
    matrix([[ 2., -1.,  0.,  0.],
            [-1.,  2., -1.,  0.],
            [ 0., -1.,  2., -1.],
            [ 0.,  0., -1.,  2.]])

    >>> # rectangular two dimensional grid 
    >>> poisson( (2,3) ).todense()
    matrix([[ 4., -1.,  0., -1.,  0.,  0.],
            [-1.,  4., -1.,  0., -1.,  0.],
            [ 0., -1.,  4.,  0.,  0., -1.],
            [-1.,  0.,  0.,  4., -1.,  0.],
            [ 0., -1.,  0., -1.,  4., -1.],
            [ 0.,  0., -1.,  0., -1.,  4.]])

    """
    grid = tuple(grid)

    N = len(grid) # grid dimension

    if N < 1 or min(grid) < 1:
        raise ValueError('invalid grid shape: %s' % str(grid))

    # create N-dimension Laplacian stencil
    stencil = np.zeros((3,) * N, dtype=dtype) 
    for i in range(N):
        stencil[ (1,)*i + (0,) + (1,)*(N-i-1) ] = -1
        stencil[ (1,)*i + (2,) + (1,)*(N-i-1) ] = -1
    stencil[ (1,)*N ] = 2*N
  
    return stencil_grid(stencil, grid, format=format)

