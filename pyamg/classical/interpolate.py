"""Classical AMG Interpolation methods"""


import numpy as np
from scipy.sparse import csr_matrix, isspmatrix_csr
from pyamg import amg_core

__all__ = ['direct_interpolation']


def direct_interpolation(A, C, splitting):
    """Create prolongator using direct interpolation

    Parameters
    ----------
    A : {csr_matrix}
        NxN matrix in CSR format
    C : {csr_matrix}
        Strength-of-Connection matrix
        Must have zero diagonal
    splitting : array
        C/F splitting stored in an array of length N

    Returns
    -------
    P : {csr_matrix}
        Prolongator using direct interpolation

    Examples
    --------
    >>> from pyamg.gallery import poisson
    >>> from pyamg.classical import direct_interpolation
    >>> import numpy as np
    >>> A = poisson((5,),format='csr')
    >>> splitting = np.array([1,0,1,0,1], dtype='intc')
    >>> P = direct_interpolation(A, A, splitting)
    >>> print P.todense()
    [[ 1.   0.   0. ]
     [ 0.5  0.5  0. ]
     [ 0.   1.   0. ]
     [ 0.   0.5  0.5]
     [ 0.   0.   1. ]]

    """
    if not isspmatrix_csr(A):
        raise TypeError('expected csr_matrix for A')

    if not isspmatrix_csr(C):
        raise TypeError('expected csr_matrix for C')

    # Interpolation weights are computed based on entries in A, but subject to
    # the sparsity pattern of C.  So, copy the entries of A into the
    # sparsity pattern of C.
    C = C.copy()
    C.data[:] = 1.0
    C = C.multiply(A)

    Pp = np.empty_like(A.indptr)

    amg_core.rs_direct_interpolation_pass1(A.shape[0],
                                           C.indptr, C.indices, splitting, Pp)

    nnz = Pp[-1]
    Pj = np.empty(nnz, dtype=Pp.dtype)
    Px = np.empty(nnz, dtype=A.dtype)

    amg_core.rs_direct_interpolation_pass2(A.shape[0],
                                           A.indptr, A.indices, A.data,
                                           C.indptr, C.indices, C.data,
                                           splitting,
                                           Pp, Pj, Px)

    return csr_matrix((Px, Pj, Pp))
