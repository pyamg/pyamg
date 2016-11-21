"""Discretizations of the Poisson problem"""
from __future__ import absolute_import

import numpy as np
import scipy as sp

from .stencil import stencil_grid

__all__ = ['poisson', 'gauge_laplacian']


def poisson(grid, spacing=None, dtype=float, format=None, type='FD'):
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
    >>> from pyamg.gallery import poisson
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

    N = len(grid)  # grid dimension

    if N < 1 or min(grid) < 1:
        raise ValueError('invalid grid shape: %s' % str(grid))

    # create N-dimension Laplacian stencil
    if type == 'FD':
        stencil = np.zeros((3,) * N, dtype=dtype)
        for i in range(N):
            stencil[(1,)*i + (0,) + (1,)*(N-i-1)] = -1
            stencil[(1,)*i + (2,) + (1,)*(N-i-1)] = -1
        stencil[(1,)*N] = 2*N

    if type == 'FE':
        stencil = -np.ones((3,) * N, dtype=dtype)
        stencil[(1,)*N] = 3**N - 1

    return stencil_grid(stencil, grid, format=format)


def gauge_laplacian(npts, spacing=1.0, beta=0.1):
    """Construct a Gauge Laplacian from Quantum Chromodynamics for
    regular 2D grids

    Note that this function is not written efficiently, but should be
    fine for N x N grids where N is in the low hundreds.

    Parameters
    ----------
    npts : {int}
        number of pts in x and y directions

    spacing : {float}
        grid spacing between points

    beta : {float}
        temperature
        Note that if beta=0, then we get the typical 5pt Laplacian stencil

    Returns
    -------
    A : {csr matrix}
        A is Hermitian positive definite for beta > 0.0
        A is Symmetric semi-definite for beta = 0.0

    Examples
    --------
    >>> from pyamg.gallery import gauge_laplacian
    >>> A = gauge_laplacian(10)

    References
    ----------
    .. [1] MacLachlan, S. and Oosterlee, C.,
       "Algebraic Multigrid Solvers for Complex-Valued Matrices",
       Vol. 30, SIAM J. Sci. Comp, 2008

    """

    # The gauge Laplacian has the same sparsity structure as a normal
    # Laplacian, so we start out with a Poisson Operator
    N = npts
    A = poisson((N, N), format='coo', dtype=complex)

    # alpha is a random function of a point's integer position
    # on a 1-D grid along the x or y direction.  e.g. the first
    # point at (0,0) would be evaluate at alpha_*[0], while the
    # last point at (N*spacing, N*spacing) would evaluate at alpha_*[-1]
    alpha_x = 1.0j * 2.0 * np.pi * beta * np.random.randn(N*N)
    alpha_y = 1.0j * 2.0 * np.pi * beta * np.random.randn(N*N)

    # Replace off diagonals of A
    for i in range(A.nnz):
        r = A.row[i]
        c = A.col[i]
        diff = np.abs(r - c)
        index = min(r, c)
        if r > c:
            s = -1.0
        else:
            s = 1.0
        if diff == 1:
            # differencing in the x-direction
            A.data[i] = -1.0 * np.exp(s * alpha_x[index])
        if diff == N:
            # differencing in the y-direction
            A.data[i] = -1.0 * np.exp(s * alpha_y[index])

    # Handle periodic BCs
    alpha_x = 1.0j * 2.0 * np.pi * beta * np.random.randn(N*N)
    alpha_y = 1.0j * 2.0 * np.pi * beta * np.random.randn(N*N)
    new_r = []
    new_c = []
    new_data = []
    new_diff = []
    for i in range(0, N):
        new_r.append(i)
        new_c.append(i + N*N - N)
        new_diff.append(N)

    for i in range(N*N - N, N*N):
        new_r.append(i)
        new_c.append(i - N*N + N)
        new_diff.append(N)

    for i in range(0, N*N-1, N):
        new_r.append(i)
        new_c.append(i + N - 1)
        new_diff.append(1)

    for i in range(N-1, N*N, N):
        new_r.append(i)
        new_c.append(i - N + 1)
        new_diff.append(1)

    for i in range(len(new_r)):
        r = new_r[i]
        c = new_c[i]
        diff = new_diff[i]
        index = min(r, c)
        if r > c:
            s = -1.0
        else:
            s = 1.0
        if diff == 1:
            # differencing in the x-direction
            new_data.append(-1.0 * np.exp(s * alpha_x[index]))
        if diff == N:
            # differencing in the y-direction
            new_data.append(-1.0 * np.exp(s * alpha_y[index]))

    # Construct Final Matrix
    data = np.hstack((A.data, np.array(new_data)))
    row = np.hstack((A.row, np.array(new_r)))
    col = np.hstack((A.col, np.array(new_c)))
    A = sp.sparse.coo_matrix((data, (row, col)), shape=(N*N, N*N)).tocsr()

    return (1.0/spacing**2)*A
