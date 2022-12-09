"""Classical AMG Interpolation methods."""


import numpy as np
from scipy.sparse import csr_matrix, isspmatrix_csr
from pyamg.strength import classical_strength_of_connection
from pyamg import amg_core

__all__ = ['direct_interpolation', 'classical_interpolation']

def direct_interpolation(A, C, splitting, theta=None, norm='min'):
    """Create prolongator using direct interpolation.
    Parameters
    ----------
    A : csr_matrix
        NxN matrix in CSR format
    C : csr_matrix
        Strength-of-Connection matrix
        Must have zero diagonal
    theta : float in [0,1), default None
        theta value defining strong connections in a classical AMG
        sense. Provide if a different SOC is used for P than for
        CF-splitting; otherwise, theta = None.
    norm : string, default 'abs'
        Norm used in redefining classical SOC. Options are 'min' and
        'abs' for CSR matrices. See strength.py for more information.
    splitting : array
        C/F splitting stored in an array of length N
    Returns
    -------
    P : csr_matrix
        Prolongator using direct interpolation
    Examples
    --------
    >>> from pyamg.gallery import poisson
    >>> from pyamg.classical.interpolate import direct_interpolation
    >>> import numpy as np
    >>> A = poisson((5,),format='csr')
    >>> splitting = np.array([1,0,1,0,1], dtype='intc')
    >>> P = direct_interpolation(A, A, splitting)
    >>> print(P.toarray())
    [[1.  0.  0. ]
     [0.5 0.5 0. ]
     [0.  1.  0. ]
     [0.  0.5 0.5]
     [0.  0.  1. ]]
    """
    if not isspmatrix_csr(A):
        raise TypeError('expected csr_matrix for A')

    if not isspmatrix_csr(C):
        raise TypeError('expected csr_matrix for C')

    if theta is not None:
        C = classical_strength_of_connection(A, theta=theta, norm=norm)
    else:
        C = C.copy()
    C.eliminate_zeros()

    # Interpolation weights are computed based on entries in A, but subject to the
    # sparsity pattern of C.  So, copy the entries of A into sparsity pattern of C.
    C.data[:] = 1.0
    C = C.multiply(A)

    P_indptr = np.empty_like(A.indptr)
    amg_core.rs_direct_interpolation_pass1(A.shape[0], C.indptr, C.indices,
                                           splitting, P_indptr)
    nnz = P_indptr[-1]
    P_indices = np.empty(nnz, dtype=P_indptr.dtype)
    P_data = np.empty(nnz, dtype=A.dtype)

    amg_core.rs_direct_interpolation_pass2(A.shape[0], A.indptr, A.indices,
                                           A.data, C.indptr, C.indices, C.data,
                                           splitting, P_indptr, P_indices, P_data)

    nc = np.sum(splitting)
    n = A.shape[0]
    return csr_matrix((P_data, P_indices, P_indptr), shape=[n,nc])


def classical_interpolation(A, C, splitting, theta=None, norm='min', modified=True):
    """Create prolongator using distance-1 classical interpolation

    Parameters
    ----------
    A : {csr_matrix}
        NxN matrix in CSR format
    C : {csr_matrix}
        Strength-of-Connection matrix
        Must have zero diagonal
    splitting : array
        C/F splitting stored in an array of length N
    theta : float in [0,1), default None
        theta value defining strong connections in a classical AMG
        sense. Provide if a different SOC is used for P than for
        CF-splitting; otherwise, theta = None.
    norm : string, default 'abs'
        Norm used in redefining classical SOC. Options are 'min' and
        'abs' for CSR matrices. See strength.py for more information.
    modified : bool, default True
        Use modified classical interpolation. More robust if RS coarsening with
        second pass is not used for CF splitting. Ignores interpolating from strong
        F-connections without a common C-neighbor.

    Returns
    -------
    P : {csr_matrix}
        Prolongator using classical interpolation; see Sec. 3 Eq. (8)
        of [0] for modified=False and Eq. (9) for modified=True.

    Examples
    --------
    >>> from pyamg.gallery import poisson
    >>> from pyamg.classical.interpolate import classical_interpolation
    >>> import numpy as np
    >>> A = poisson((5,),format='csr')
    >>> splitting = np.array([1,0,1,0,1], dtype='intc')
    >>> P = classical_interpolation(A, A, splitting, 0.25)
    >>> print(P.todense())
    [[ 1.   0.   0. ]
     [ 0.5  0.5  0. ]
     [ 0.   1.   0. ]
     [ 0.   0.5  0.5]
     [ 0.   0.   1. ]]
    """
    if not isspmatrix_csr(A):
        raise TypeError('expected csr_matrix for A')

    if not isspmatrix_csr(C):
        raise TypeError('Expected csr_matrix SOC matrix, C.')

    nc = np.sum(splitting)
    n = A.shape[0]

    if theta is not None:
        C = classical_strength_of_connection(A, theta=theta, norm=norm)
    else:
        C = C.copy()

    # Use modified classical interpolation by ignoring strong F-connections that do
    # not have a common C-point.
    if modified:
        amg_core.remove_strong_FF_connections(A.shape[0], C.indptr, C.indices, C.data, splitting)
    C.eliminate_zeros()

    # Interpolation weights are computed based on entries in A, but subject to
    # the sparsity pattern of C.  So, copy the entries of A into the
    # sparsity pattern of C.
    C.data[:] = 1.0
    C = C.multiply(A)

    P_indptr = np.empty_like(A.indptr)
    amg_core.rs_classical_interpolation_pass1(A.shape[0], C.indptr,
                                             C.indices, splitting, P_indptr)
    nnz = P_indptr[-1]
    P_indices = np.empty(nnz, dtype=P_indptr.dtype)
    P_data = np.empty(nnz, dtype=A.dtype)

    amg_core.rs_classical_interpolation_pass2(A.shape[0], A.indptr, A.indices,
                                             A.data, C.indptr, C.indices,
                                             C.data, splitting, P_indptr,
                                             P_indices, P_data, modified)

    return csr_matrix((P_data, P_indices, P_indptr), shape=[n,nc])
