"""Random sparse matrices"""

import numpy as np
import scipy as sp

__all__ = ['sprand']

# TODO add sprandn


def _rand_sparse(m, n, density, format='csr'):
    """Helper function for sprand, sprandn"""

    nnz = max(min(int(m*n*density), m*n), 0)

    row = np.random.randint(low=0, high=m-1, size=nnz)
    col = np.random.randint(low=0, high=n-1, size=nnz)
    data = np.ones(nnz, dtype=float)

    # duplicate (i,j) entries will be summed together
    return sp.sparse.csr_matrix((data, (row, col)), shape=(m, n))


def sprand(m, n, density, format='csr'):
    """Returns a random sparse matrix.

    Parameters
    ----------
    m, n : int
        shape of the result
    density : float
        target a matrix with nnz(A) = m*n*density, 0<=density<=1
    format : string
        sparse matrix format to return, e.g. 'csr', 'coo', etc.

    Returns
    -------
    A : sparse matrix
        m x n sparse matrix

    Examples
    --------
    >>> from pyamg.gallery import sprand
    >>> A = sprand(5,5,3/5.0)

    """
    m, n = int(m), int(n)

    # get sparsity pattern
    A = _rand_sparse(m, n, density, format='csr')

    # replace data with random values
    A.data = sp.rand(A.nnz)

    return A.asformat(format)


# currently returns positive semi-definite matrices
# def sprand_spd(m, n, density, a=1.0, b=2.0, format='csr'):
#    """Returns a random sparse, symmetric positive definite matrix
#
#    Parameters
#    ----------
#    n : int
#        shape of the result
#    density : float
#        target a matrix with nnz(A) = m*n*density, 0<=density<=1
#    a,b : float
#        eigenvalues of the result will lie in the range [a,b]
#    format : string
#        sparse matrix format to return, e.g. "csr", "coo", etc.
#
#    Returns
#    -------
#    A : sparse, SPD matrix
#        n x n sparse matrix with eigenvalues in the interval [a,b]
#
#    Examples
#    --------
#
#    See Also
#    --------
#    pyamg.classical.cr.binormalize
#
#    """
#    # get sparsity pattern
#    A = _rand_sparse(n, n, density, format='csr')
#
#    A.data = sp.rand(A.nnz)
#
#    A = sp.sparse.tril(A, -1)
#
#    A = A + A.T
#
#    d = np.array(A.sum(axis=1)).ravel()
#
#    from pyamg.util.utils import symmetric_rescaling, diag_sparse
#
#    D = diag_sparse(d)
#    A = D - A
#
#    # A now has zero row sums
#    D_sqrt,D_sqrt_inv,A = symmetric_rescaling(A)
#
#    # A now has unit diagonals
#
#    return A.asformat(format)
