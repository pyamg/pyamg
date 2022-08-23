"""Classical AMG Interpolation methods."""
import numpy as np
from scipy.sparse import csr_matrix, isspmatrix_csr
from pyamg.strength import classical_strength_of_connection
from pyamg import amg_core

__all__ = ['direct_interpolation', 'standard_interpolation',
           'distance_two_interpolation']

def direct_interpolation(A, C, splitting, theta=None, norm='min'):
    """Create prolongator using direct interpolation.

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
        C0 = classical_strength_of_connection(A, theta=theta, norm=norm)
    else:
        # BS - had this in my code, can't remember why; presumably need C later?
        # C0 = C.copy()
        C0 = C
    C0.eliminate_zeros()

    # Interpolation weights are computed based on entries in A, but subject to the
    # sparsity pattern of C.  So, copy the entries of A into sparsity pattern of C.
    C0.data[:] = 1.0
    C0 = C0.multiply(A)

    P_indptr = np.empty_like(A.indptr)
    amg_core.rs_direct_interpolation_pass1(A.shape[0], C0.indptr, C0.indices, 
                                           splitting, P_indptr)
    nnz = P_indptr[-1]
    P_colinds = np.empty(nnz, dtype=P_indptr.dtype)
    P_data = np.empty(nnz, dtype=A.dtype)

    amg_core.rs_direct_interpolation_pass2(A.shape[0], A.indptr, A.indices,
                                           A.data, C0.indptr, C0.indices, C0.data,
                                           splitting, P_indptr, P_colinds, P_data)

    nc = np.sum(splitting)
    n = A.shape[0]
    return csr_matrix((P_data, P_colinds, P_indptr), shape=[n,nc])



def standard_interpolation(A, C, splitting, theta=None, norm='min', modified=True):
    """Create prolongator using distance-1 standard/classical interpolation

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
        raise TypeError('Expected csr_matrix SOC matrix, C.')

    nc = np.sum(splitting)
    n = A.shape[0]

    if theta is not None:
        C0 = classical_strength_of_connection(A, theta=theta, norm=norm)
    else:
        # BS - had this in my code, can't remember why; presumably need C later?
        # C0 = C.copy()
        C0 = C

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

    return csr_matrix((P_data, P_colinds, P_indptr), shape=[n,nc])



def distance_two_interpolation(A, C, splitting, theta=None, norm='min', plus_i=True):
    """Create prolongator using distance-two AMG interpolation (extended+i interpolaton).

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
    plus_i : bool, default True
        Use "Extended+i" interpolation from [0] as opposed to "Extended"
        interpolation. Typically gives better interpolation with minimal
        added expense.

    Returns
    -------
    P : {csr_matrix}
        Prolongator using standard interpolation

    References
    ----------
    [0] "Distance-Two Interpolation for Parallel Algebraic Multigrid,"
       H. De Sterck, R. Falgout, J. Nolting, U. M. Yang, (2007).

    Examples
    --------


    """
    if not isspmatrix_csr(C):
        raise TypeError('Expected csr_matrix SOC matrix, C.')

    nc = np.sum(splitting)
    n = A.shape[0]

    if theta is not None:
        C0 = classical_strength_of_connection(A, theta=theta, norm=norm)
    else:
        # BS - had this in my code, can't remember why; presumably need C later?
        # C0 = C.copy()
        C0 = C
    C0.eliminate_zeros()

    # Interpolation weights are computed based on entries in A, but subject to
    # the sparsity pattern of C.  So, copy the entries of A into the
    # sparsity pattern of C.
    C0.data[:] = 1.0
    C0 = C0.multiply(A)

    P_indptr = np.empty_like(A.indptr)
    amg_core.distance_two_amg_interpolation_pass1(A.shape[0], C0.indptr,
                                                  C0.indices, splitting, P_indptr)
    nnz = P_indptr[-1]
    P_colinds = np.empty(nnz, dtype=P_indptr.dtype)
    P_data = np.empty(nnz, dtype=A.dtype)
    if plus_i:
        amg_core.extended_plusi_interpolation_pass2(A.shape[0], A.indptr, A.indices,
                                                    A.data, C0.indptr, C0.indices,
                                                    C0.data, splitting, P_indptr,
                                                    P_colinds, P_data)
    else:
        amg_core.extended_interpolation_pass2(A.shape[0], A.indptr, A.indices,
                                              A.data, C0.indptr, C0.indices,
                                              C0.data, splitting, P_indptr,
                                              P_colinds, P_data)

    return csr_matrix((P_data, P_colinds, P_indptr), shape=[n,nc])

