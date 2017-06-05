"""Classical AMG Interpolation methods"""


__docformat__ = "restructuredtext en"

import numpy as np
from scipy.sparse import csr_matrix, bsr_matrix, isspmatrix_csr, isspmatrix_bsr
from pyamg import amg_core
from pyamg.util.utils import UnAmal


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
    if not isspmatrix_csr(C):
        raise TypeError('Expected csr_matrix SOC matrix, C.')

    if isspmatrix_bsr(A):
        C0 = UnAmal(C, A.blocksize[0], A.blocksize[1])
        A_temp = A.tocsr()
    else:
        C0 = C.copy()   # Copy strength matrix
        A_temp = A      # Reference matrix A  

    # Interpolation weights are computed based on entries in A, but subject to the
    # sparsity pattern of C.  So, copy the entries of A into sparsity pattern of C.
    C0.eliminate_zeros()
    C0.data[:] = 1.0
    C0 = C0.multiply(A)

    P_indptr = np.empty_like(temp_A.indptr)
    amg_core.rs_direct_interpolation_pass1(temp_A.shape[0], C0.indptr, C0.indices, 
                                           splitting, P_indptr)

    nnz = P_indptr[-1]
    P_colinds = np.empty(nnz, dtype=P_indptr.dtype)
    P_data = np.empty(nnz, dtype=A.dtype)

    amg_core.rs_direct_interpolation_pass2(temp_A.shape[0], temp_A.indptr, temp_A.indices,
                                           temp_A.data, C0.indptr, C0.indices, C0.data,
                                           splitting, P_indptr, P_colinds, P_data)

    if isspmatrix_bsr(A):
        return bsr_matrix((P_data, P_colinds, P_indptr), blocksize=A.blocksize)
    else:    
        return csr_matrix((P_data, P_colinds, P_indptr))


def standard_interpolation(A, C, splitting, theta=None, norm='min', modified=True, cost=[0]):
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
    norm : string, default 'abs'
        Norm used in redefining classical SOC. Options are 'min' and 'abs' for CSR matrices,
        and 'min', 'abs', and 'fro' for BSR matrices. See strength.py for more information.
    modified : bool, default True
        Use modified classical interpolation. More robust if RS coarsening with second
        pass is not used for CF splitting. Ignores interpolating from strong F-connections
        without a common C-neighbor.

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
    if not isspmatrix_csr(C):
        raise TypeError('Expected csr_matrix SOC matrix, C.')

    if isspmatrix_bsr(A):
        A_temp = A.tocsr()
        if theta is not None:
            C0 = classical_strength_of_connection(A, theta=theta, norm=norm, cost=cost)
            C0 = UnAmal(C0, A.blocksize[0], A.blocksize[1])
        else:
            C0 = UnAmal(C, A.blocksize[0], A.blocksize[1])
    else:
        A_temp = A      # Reference matrix A  
        if theta is not None:
            C0 = classical_strength_of_connection(A, theta=theta, norm=norm, cost=cost)
        else:
            C0 = C.copy()   # Copy strength matrix

    # Use modified standard interpolation by ignoring strong F-connections that do
    # not have a common C-point.
    if modified:
        amg_core.remove_strong_FF_connections(temp_A.shape[0], C0.indptr, C0.indices, C0.data, splitting)
    C0.eliminate_zeros()

    # Interpolation weights are computed based on entries in A, but subject to
    # the sparsity pattern of C.  So, copy the entries of A into the
    # sparsity pattern of C.
    C0.data[:] = 1.0
    C0 = C0.multiply(temp_A)

    P_indptr = np.empty_like(temp_A.indptr)
    amg_core.rs_standard_interpolation_pass1(temp_A.shape[0], C0.indptr,
                                             C0.indices, splitting, P_indptr)

    nnz = P_indptr[-1]
    P_colinds = np.empty(nnz, dtype=P_indptr.dtype)
    P_data = np.empty(nnz, dtype=temp_A.dtype)

    if modified:
        amg_core.mod_standard_interpolation_pass2(temp_A.shape[0], temp_A.indptr, temp_A.indices,
                                                  temp_A.data, C0.indptr, C0.indices,
                                                  C0.data, splitting, P_indptr,
                                                  P_colinds, P_data)
    else:
        amg_core.rs_standard_interpolation_pass2(temp_A.shape[0], temp_A.indptr, temp_A.indices,
                                                 temp_A.data, C0.indptr, C0.indices,
                                                 C0.data, splitting, P_indptr,
                                                 P_colinds, P_data)

    if isspmatrix_bsr(A):
        return bsr_matrix((P_data, P_colinds, P_indptr), blocksize=A.blocksize)
    else:    
        return csr_matrix((P_data, P_colinds, P_indptr))




# TODO : Add distance-two interp to python interface




