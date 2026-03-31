"""Construct sparse matrix from a local stencil."""
# pylint: disable=redefined-builtin

import numpy as np
from scipy import sparse


def stencil_grid(S, grid, dtype=None, format=None):
    """Construct a sparse matrix form a local matrix stencil.

    Parameters
    ----------
    S : ndarray
        Matrix stencil stored in n-d array.
    grid : tuple
        Tuple containing the grid dimensions.
    dtype : dtype
        Data type of the result.
    format : str
        Sparse array format to return, e.g. "csr", "coo", etc.

    Returns
    -------
    sparray
        Sparse array which represents the operator given by applying
        stencil S at each vertex of a regular grid with given dimensions.

    Notes
    -----
    The grid vertices are enumerated as ``arange(prod(grid)).reshape(grid)``.
    This implies that the last grid dimension cycles fastest, while the
    first dimension cycles slowest.  For example, if ``grid=(2,3)`` then the
    grid vertices are ordered as (0,0), (0,1), (0,2), (1,0), (1,1), (1,2).

    This coincides with the ordering used by the NumPy functions
    :meth:`numpy.ndenumerate` and :meth:`numpy.mgrid`.

    Examples
    --------
    >>> from pyamg.gallery import stencil_grid
    >>> stencil = [-1,2,-1]  # 1D Poisson stencil
    >>> grid = (5,)          # 1D grid with 5 vertices
    >>> A = stencil_grid(stencil, grid, dtype=float, format='csr')
    >>> A.toarray()
    array([[ 2., -1.,  0.,  0.,  0.],
           [-1.,  2., -1.,  0.,  0.],
           [ 0., -1.,  2., -1.,  0.],
           [ 0.,  0., -1.,  2., -1.],
           [ 0.,  0.,  0., -1.,  2.]])

    >>> stencil = [[0,-1,0],[-1,4,-1],[0,-1,0]] # 2D Poisson stencil
    >>> grid = (3,3)                            # 2D grid with shape 3x3
    >>> A = stencil_grid(stencil, grid, dtype=float, format='csr')
    >>> A.toarray()
    array([[ 4., -1.,  0., -1.,  0.,  0.,  0.,  0.,  0.],
           [-1.,  4., -1.,  0., -1.,  0.,  0.,  0.,  0.],
           [ 0., -1.,  4.,  0.,  0., -1.,  0.,  0.,  0.],
           [-1.,  0.,  0.,  4., -1.,  0., -1.,  0.,  0.],
           [ 0., -1.,  0., -1.,  4., -1.,  0., -1.,  0.],
           [ 0.,  0., -1.,  0., -1.,  4.,  0.,  0., -1.],
           [ 0.,  0.,  0., -1.,  0.,  0.,  4., -1.,  0.],
           [ 0.,  0.,  0.,  0., -1.,  0., -1.,  4., -1.],
           [ 0.,  0.,  0.,  0.,  0., -1.,  0., -1.,  4.]])

    """
    S = np.asarray(S, dtype=dtype)
    grid = tuple(grid)

    if not (np.asarray(S.shape) % 2 == 1).all():
        raise ValueError('all stencil dimensions must be odd')

    if len(grid) != np.ndim(S):
        raise ValueError('stencil dimension must equal number of grid\
                          dimensions')

    if min(grid) < 1:
        raise ValueError('grid dimensions must be positive')

    N_v = np.prod(grid)  # number of vertices in the mesh
    N_s = (S != 0).sum()    # number of nonzero stencil entries

    # diagonal offsets
    diags = np.zeros(N_s, dtype=int)

    # compute index offset of each dof within the stencil
    strides = np.cumprod([1] + list(reversed(grid)))[:-1]  # noqa: RUF005
    indices = tuple(i.astype(np.int32) for i in S.nonzero())
    for i, s in zip(indices, S.shape, strict=False):
        i -= s // 2
        # i = (i - s) // 2
        # i = i // 2
        # i = i - (s // 2)
    for stride, coords in zip(strides, reversed(indices), strict=False):
        diags += stride * coords

    data = S[S != 0].repeat(N_v).reshape(N_s, N_v)

    indices = np.vstack(indices).T

    # zero boundary connections
    for index, diag in zip(indices, data, strict=False):
        diag = diag.reshape(grid)
        for n, i in enumerate(index):
            if i > 0:
                s = [slice(None)] * len(grid)
                s[n] = slice(0, i)
                s = tuple(s)
                diag[s] = 0
            elif i < 0:
                s = [slice(None)]*len(grid)
                s[n] = slice(i, None)
                s = tuple(s)
                diag[s] = 0

    # remove diagonals that lie outside matrix
    mask = abs(diags) < N_v
    if not mask.all():
        diags = diags[mask]
        data = data[mask]

    # sum duplicate diagonals
    if len(np.unique(diags)) != len(diags):
        new_diags = np.unique(diags)
        new_data = np.zeros((len(new_diags), data.shape[1]),
                            dtype=data.dtype)

        for dia, dat in zip(diags, data, strict=False):
            n = np.searchsorted(new_diags, dia)
            new_data[n, :] += dat

        diags = new_diags
        data = new_data

    return sparse.dia_array((data, diags),
                             shape=(N_v, N_v)).asformat(format)


def stencil_grid_bc(stencil, offsets, grid_shape, dtype=np.double, boundary='periodic'):
    """
    Create sparse d-D matrix from a stencil.

    Parameters
    ----------
    stencil : ndarray
        matrix stencil stored in d-D array
    offsets: list
        corresponding diagonal offsets. 0 corresponds to the middle of the stencil.
    grid_shape : tuple
        tuple containing the d grid dimensions
    dtype : numpy dtype
        data type of the result
    boundary : string
        type of boundary condition.
            'dirichlet': stencil is truncated to zero outside domain.
            'periodic': periodic boundary conditions.

    Returns
    -------
    A : csr_matrix
        n x n sparse matrix with vals[i, j] on diagonal offsets[i, j]

    Examples
    --------
    >>> stencil_grid([2, -1, -1], [(0, ), (-1, ), (1, )], (6, ),
                     boundary='dirichlet').toarray()
    array([[ 2., -1.,  0.,  0.,  0., 0.],
           [-1.,  2., -1.,  0.,  0.,  0.],
           [ 0., -1.,  2., -1.,  0.,  0.],
           [ 0.,  0., -1.,  2., -1.,  0.],
           [ 0.,  0.,  0., -1.,  2., -1.],
           [ 0.,  0.,  0.,  0., -1.,  2.]])
    >>> stencil_grid([4, -1, -1, -1, -1], [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)],
                     (4, 4), boundary='periodic').toarray()
    array([[ 4., -1.,  0., -1., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -1.,  0.,  0.,  0.],
           [-1.,  4., -1.,  0.,  0., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -1.,  0.,  0.],
           [ 0., -1.,  4., -1.,  0.,  0., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -1.,  0.],
           [-1.,  0., -1.,  4.,  0.,  0.,  0., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -1.],
           [-1.,  0.,  0.,  0.,  4., -1.,  0., -1., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
           [ 0., -1.,  0.,  0., -1.,  4., -1.,  0.,  0., -1.,  0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0., -1.,  0.,  0., -1.,  4., -1.,  0.,  0., -1.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0., -1., -1.,  0., -1.,  4.,  0.,  0.,  0., -1.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0., -1.,  0.,  0.,  0.,  4., -1.,  0., -1., -1.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0., -1.,  0.,  0., -1.,  4., -1.,  0.,  0., -1.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0., -1.,  0.,  0., -1.,  4., -1.,  0.,  0., -1.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0., -1., -1.,  0., -1.,  4.,  0.,  0.,  0., -1.],
           [-1.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -1.,  0.,  0.,  0.,  4., -1.,  0., -1.],
           [ 0., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -1.,  0.,  0., -1.,  4., -1.,  0.],
           [ 0.,  0., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -1.,  0.,  0., -1.,  4., -1.],
           [ 0.,  0.,  0., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -1., -1.,  0., -1.,  4.]])
    """
    grid_size = np.prod(grid_shape)

    # Gridpoint coordinates.
    x = np.unravel_index(np.arange(grid_size, dtype=int), grid_shape)
    row = np.concatenate(tuple(np.arange(grid_size) for _ in range(len(stencil))))
    data = np.concatenate(tuple([v] * grid_size for v in stencil))

    if boundary == 'periodic':
        col_sub = np.array([np.concatenate(tuple(np.mod(xd + offset[d], grid_shape[d])
                                                 for offset in offsets))
                            for d, xd in enumerate(x)]).T
    elif boundary == 'dirichlet':
        col_sub = np.array([np.concatenate(tuple(xd + offset[d]
                                                 for offset in offsets))
                            for d, xd in enumerate(x)]).T
        in_domain = np.logical_and.reduce(np.array([
            (0 <= col_sub[:, d]) & (col_sub[:, d] < grid_shape[d])
            for d in range(len(grid_shape))]))
        row, col_sub, data = row[in_domain], col_sub[in_domain], data[in_domain]
    else:
        raise Exception(f'Unsupported boundary conditions {boundary}')

    col = np.array([np.ravel_multi_index(sub, grid_shape) for sub in col_sub])

    return sparse.csr_matrix((data, (row, col)), shape=(grid_size, grid_size), dtype=dtype)
