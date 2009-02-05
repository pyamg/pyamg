from numpy.random import random_integers
from scipy import rand, randn, ones, array
from scipy.sparse import csr_matrix, tril
from pyamg.util.utils import symmetric_rescaling, diag_sparse

def _rand_sparse(grid, density, format='csr'):
    """Helper function for sprand, sprandn
    """
    m = grid[0]
    n = grid[1]

    nnz = max( min( int(m*n*density), m*n), 0)

    row  = random_integers(low=0, high=m-1, size=nnz)
    col  = random_integers(low=0, high=n-1, size=nnz)
    data = ones(nnz, dtype=float)

    A = csr_matrix( (data,(row,col)), shape=(m,n) )

    # duplicate (i,j) entries will be summed together
    return csr_matrix( (data,(row,col)), shape=(m,n) )

def sprand(grid, density, format='csr'):
    """Returns a random sparse matrix.

    Parameters
    ----------
    grid : tuple of integers
        grid dimensions e.g. (100,100)
    density : float
        target a matrix with nnz(A) = m*n*density, 0<=density<=1
    format : string
        sparse matrix format to return, e.g. "csr", "coo", etc.

    Returns
    -------
    A : sparse matrix
        m x n sparse matrix

    Examples
    --------
    >>>> print sprand((5,5),3/5.0).todense()
    [[ 0.55331722  0.          0.35156318  0.68261756  0.62756243]
     [ 0.          0.          0.          0.          0.        ]
     [ 0.          0.97096491  0.          0.          0.45973418]
     [ 0.          0.41185779  0.          0.40211105  0.        ]
     [ 0.06545295  0.          0.32022103  0.75251759  0.        ]]

    """
    grid = tuple(grid)

    dim = len(grid) # grid dimension

    if(dim != 2 or min(grid) < 1):
        raise ValueError('invalid grid shape: %s' % str(grid))

    A = _rand_sparse(grid, density, format='csr')

    A.data = rand(A.nnz)
    return A.asformat(format)

def sprand_spd(grid, density, format='csr'):
    """Returns a random sparse, symmetric semi-positive definite matrix, with
    row sum zero

    Parameters
    ----------
    grid : tuple of integers
        grid dimensions e.g. (100,100)
    density : float
        target a matrix with nnz(A) = m*n*density, 0<=density<=1
    format : string
        sparse matrix format to return, e.g. "csr", "coo", etc.

    Returns
    -------
    A : sparse, s.p.d. matrix
        m x n sparse matrix with positive diagonal and negative off-diagonals
        with row sum 0.

    Examples
    --------
    >>> print sprand((5,5),3/5.0).todense()
    [[ 2.28052355  0.         -0.72223852 -0.73750262 -0.82078241]
     [ 0.          0.74648788 -0.52911585  0.         -0.21737203]
     [-0.72223852 -0.52911585  1.25135436  0.          0.        ]
     [-0.73750262  0.          0.          0.73750262  0.        ]
     [-0.82078241 -0.21737203  0.          0.          1.03815444]]

     See Also
     --------
     pyamg.classical.cr.binormalize

    """
    grid = tuple(grid)

    dim = len(grid) # grid dimension

    if(dim != 2 or min(grid) < 1):
        raise ValueError('invalid grid shape: %s' % str(grid))

    A = _rand_sparse(grid, density, format='csr')
    A.data = -1.0*rand(A.nnz)

    A = tril(A,-1)

    A = A + A.T

    d = -array(A.sum(axis=1)).ravel()

    D = diag_sparse(d)
    A = A + D

    # TODO : will a rescaling preserve spd here?
    #D_sqrt,D_sqrt_inv,A = symmetric_rescaling(A)
    return A.asformat(format)
