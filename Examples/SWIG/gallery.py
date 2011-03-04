import numpy
import scipy
import scipy.sparse
"""Discretizations of the Poisson problem"""

__docformat__ = "restructuredtext en"

__all__ = ['poisson','stencil_grid']

def stencil_grid(S, grid, dtype=None, format=None):
    """Construct a sparse matrix form a local matrix stencil 
    
    Parameters
    ----------
    S : ndarray
        matrix stencil stored in rank N array
    grid : tuple
        tuple containing the N grid dimensions
    dtype :
        data type of the result
    format : string
        sparse matrix format to return, e.g. "csr", "coo", etc.

    Returns
    -------
    A : sparse matrix
        Sparse matrix which represents the operator given by applying
        stencil S at each vertex of a regular grid with given dimensions.

    Notes
    -----
    The grid vertices are enumerated as arange(prod(grid)).reshape(grid).
    This implies that the last grid dimension cycles fastest, while the 
    first dimension cycles slowest.  For example, if grid=(2,3) then the
    grid vertices are ordered as (0,0), (0,1), (0,2), (1,0), (1,1), (1,2).

    This coincides with the ordering used by the NumPy functions 
    ndenumerate() and mgrid().

    Examples
    --------
    >>> stencil = [-1,2,-1]  # 1D Poisson stencil
    >>> grid = (5,)          # 1D grid with 5 vertices
    >>> A = stencil_grid(stencil, grid, dtype=float, format='csr')   
    >>> A.todense()
    matrix([[ 2., -1.,  0.,  0.,  0.],
            [-1.,  2., -1.,  0.,  0.],
            [ 0., -1.,  2., -1.,  0.],
            [ 0.,  0., -1.,  2., -1.],
            [ 0.,  0.,  0., -1.,  2.]])
    
    >>> stencil = [[0,-1,0],[-1,4,-1],[0,-1,0]] # 2D Poisson stencil
    >>> grid = (3,3)                            # 2D grid with shape 3x3
    >>> A = stencil_grid(stencil, grid, dtype=float, format='csr')   
    >>> A.todense()
    matrix([[ 4., -1.,  0., -1.,  0.,  0.,  0.,  0.,  0.],
            [-1.,  4., -1.,  0., -1.,  0.,  0.,  0.,  0.],
            [ 0., -1.,  4.,  0.,  0., -1.,  0.,  0.,  0.],
            [-1.,  0.,  0.,  4., -1.,  0., -1.,  0.,  0.],
            [ 0., -1.,  0., -1.,  4., -1.,  0., -1.,  0.],
            [ 0.,  0., -1.,  0., -1.,  4.,  0.,  0., -1.],
            [ 0.,  0.,  0., -1.,  0.,  0.,  4., -1.,  0.],
            [ 0.,  0.,  0.,  0., -1.,  0., -1.,  4., -1.],
            [ 0.,  0.,  0.,  0.,  0., -1.,  0., -1.,  4.]])
    
    """

    S = numpy.asarray(S, dtype=dtype)
    grid = tuple(grid)

    if not (numpy.asarray(S.shape) % 2 == 1).all():
        raise ValueError('all stencil dimensions must be odd')
    
    if len(grid) != numpy.rank(S):
        raise ValueError('stencil rank must equal number of grid dimensions')
    
    if min(grid) < 1:
        raise ValueError('grid dimensions must be positive')
    
    N_v = numpy.prod(grid)  # number of vertices in the mesh
    N_s = (S != 0).sum()    # number of nonzero stencil entries

    # diagonal offsets 
    diags = numpy.zeros(N_s, dtype=int)  

    # compute index offset of each DoF within the stencil
    strides = numpy.cumprod( [1] + list(reversed(grid)) )[:-1]
    indices = S.nonzero()
    for i,s in zip(indices,S.shape):
        i -= s // 2
    for stride,coords in zip(strides, reversed(indices)):
        diags += stride * coords

    data = S[ S != 0 ].repeat(N_v).reshape(N_s,N_v)

    indices = numpy.vstack(indices).T

    # zero boundary connections
    for index,diag in zip(indices,data):
        diag = diag.reshape(grid)
        for n,i in enumerate(index):
            if i > 0:
                s = [ slice(None) ]*len(grid)
                s[n] = slice(0,i)
                diag[s] = 0
            elif i < 0:
                s = [ slice(None) ]*len(grid)
                s[n] = slice(i,None)
                diag[s] = 0

    # remove diagonals that lie outside matrix
    mask = abs(diags) < N_v
    if not mask.all():
        diags = diags[mask]
        data  = data[mask]
    
    # sum duplicate diagonals
    if len(numpy.unique(diags)) != len(diags):
        new_diags = numpy.unique(diags)
        new_data  = numpy.zeros( (len(new_diags),data.shape[1]), dtype=data.dtype)

        for dia,dat in zip(diags,data):
            n = numpy.searchsorted(new_diags,dia)
            new_data[n,:] += dat
        
        diags = new_diags
        data  = new_data

    return scipy.sparse.dia_matrix((data,diags), shape=(N_v,N_v)).asformat(format)

def poisson(grid, spacing=None, dtype=float, format=None):
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
    stencil = numpy.zeros((3,) * N, dtype=dtype) 
    for i in range(N):
        stencil[ (1,)*i + (0,) + (1,)*(N-i-1) ] = -1
        stencil[ (1,)*i + (2,) + (1,)*(N-i-1) ] = -1
    stencil[ (1,)*N ] = 2*N
  
    return stencil_grid(stencil, grid, format=format)
