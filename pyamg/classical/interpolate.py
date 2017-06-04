"""Classical AMG Interpolation methods"""


__docformat__ = "restructuredtext en"

import numpy as np
from scipy.sparse import csr_matrix, isspmatrix_csr
from pyamg import amg_core

__all__ = ['direct_interpolation', 'standard_interpolation']


def direct_interpolation(A, C, splitting, cost=[0]):
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

    P_indptr = np.empty_like(A.indptr)

    amg_core.rs_direct_interpolation_pass1(A.shape[0],
                                           C.indptr, C.indices, splitting, P_indptr)

    nnz = P_indptr[-1]
    P_colinds = np.empty(nnz, dtype=P_indptr.dtype)
    P_data = np.empty(nnz, dtype=A.dtype)

    amg_core.rs_direct_interpolation_pass2(A.shape[0],
                                           A.indptr, A.indices, A.data,
                                           C.indptr, C.indices, C.data,
                                           splitting,
                                           P_indptr, P_colinds, P_data)

    return csr_matrix((P_data, P_colinds, P_indptr))


def standard_interpolation(A, C, splitting, theta=None, norm=None, modified=True, cost=[0]):
    """Create prolongator using standard interpolation

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
        theta value defining strong connections in a classical AMG sense. Provide if
        different SOC used for P than for CF-splitting; otherwise, theta = None. 
    norm : string
        norm used in redefining classical SOC -- TODO : list options
    modified : bool, default True
        Use modified classical interpolation. More robust if RS coarsening with second
        pass is not used for CF splitting.
    distance_two : bool, default False

    Returns
    -------
    P : {csr_matrix}
        Prolongator using standard interpolation

    Examples
    --------
    >>> from pyamg.gallery import poisson
    >>> from pyamg.classical import standard_interpolation
    >>> import numpy as np
    >>> A = poisson((5,),format='csr')
    >>> splitting = np.array([1,0,1,0,1], dtype='intc')
    >>> P = standard_interpolation(A, A, splitting)
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

    if theta is not None:
        if norm is None:
            C0 = classical_strength_of_connection(A, theta=theta, block=None, norm='min', cost=cost)
        else:
            C0 = classical_strength_of_connection(A, theta=theta, block=None, norm=norm, cost=cost)
    else:
        C0 = C.copy()

    # Use modified standard interpolation by ignoring strong F-connections that do
    # not have a common C-point.
    if modified:
        amg_core.remove_strong_FF_connections(A.shape[0], C0.indptr, C0.indices, C0.data, splitting)
    C0.eliminate_zeros()

    # Interpolation weights are computed based on entries in A, but subject to
    # the sparsity pattern of C.  So, copy the entries of A into the
    # sparsity pattern of C.
    C0.data[:] = 1.0
    C0 = C0.multiply(A)

    P_indptr = np.empty_like(A.indptr)
    amg_core.rs_standard_interpolation_pass1(A.shape[0], C0.indptr,
                                             C0.indices, splitting, P_indptr)

    nnz = P_indptr[-1]
    P_colinds = np.empty(nnz, dtype=P_indptr.dtype)
    P_data = np.empty(nnz, dtype=A.dtype)

    if modified:
        amg_core.mod_standard_interpolation_pass2(A.shape[0], A.indptr, A.indices,
                                                  A.data, C0.indptr, C0.indices,
                                                  C0.data, splitting, P_indptr,
                                                  P_colinds, P_data)
    else:
        amg_core.rs_standard_interpolation_pass2(A.shape[0], A.indptr, A.indices,
                                                 A.data, C0.indptr, C0.indices,
                                                 C0.data, splitting, P_indptr,
                                                 P_colinds, P_data)

    return  csr_matrix((P_data, P_colinds, P_indptr))
