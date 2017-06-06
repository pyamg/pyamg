"""Classical AMG Interpolation methods"""


__docformat__ = "restructuredtext en"

import numpy as np
from scipy.sparse import csr_matrix, bsr_matrix, isspmatrix_csr, isspmatrix_bsr
from pyamg import amg_core
from pyamg.util.utils import UnAmal
from pyamg.strength import classical_strength_of_connection


__all__ = ['direct_interpolation', 'standard_interpolation',
           'distance_two_interpolation', 'injection_interpolation',
           'one_point_interpolation']


def direct_interpolation(A, C, splitting, theta=None, norm='min', cost=[0]):
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
    theta : float in [0,1), default None
        theta value defining strong connections in a classical AMG sense. Provide if
        different SOC used for P than for CF-splitting; otherwise, theta = None. 
    norm : string, default 'abs'
        Norm used in redefining classical SOC. Options are 'min' and 'abs' for CSR matrices,
        and 'min', 'abs', and 'fro' for BSR matrices. See strength.py for more information.

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

    # Block BSR format. Transfer A to CSR and the splitting and SOC matrix to have
    # DOFs corresponding to CSR A
    if isspmatrix_bsr(A):
        temp_A = A.tocsr()
        temp_A.eliminate_zeros()
        splitting0 = splitting * np.ones((A.blocksize[0],1), dtype='intc')
        splitting0 = np.reshape(splitting0, (np.prod(splitting0.shape),), order='F')
        if theta is not None:
            C0 = classical_strength_of_connection(A, theta=theta, norm=norm, cost=cost)
            C0 = UnAmal(C0, A.blocksize[0], A.blocksize[1])
        else:
            C0 = UnAmal(C, A.blocksize[0], A.blocksize[1])
        C0 = C0.tocsr()
        C0.eliminate_zeros()

        # Interpolation weights are computed based on entries in A, but subject to the
        # sparsity pattern of C.  So, copy the entries of A into sparsity pattern of C.
        C0.data[:] = 1.0
        C0 = C0.multiply(temp_A)

        P_indptr = np.empty_like(temp_A.indptr)
        amg_core.rs_direct_interpolation_pass1(temp_A.shape[0], C0.indptr, C0.indices, 
                                               splitting0, P_indptr)

        nnz = P_indptr[-1]
        P_colinds = np.empty(nnz, dtype=P_indptr.dtype)
        P_data = np.empty(nnz, dtype=A.dtype)

        amg_core.rs_direct_interpolation_pass2(temp_A.shape[0], temp_A.indptr, temp_A.indices,
                                               temp_A.data, C0.indptr, C0.indices, C0.data,
                                               splitting0, P_indptr, P_colinds, P_data)

        nc = np.sum(splitting0)
        n = A.shape[0]
        P = csr_matrix((P_data, P_colinds, P_indptr), shape=[n,nc])
        return P.tobsr(blocksize=A.blocksize)

    # CSR format
    else:
        if theta is not None:
            C0 = classical_strength_of_connection(A, theta=theta, norm=norm, cost=cost)
        else:
            C0 = C.copy()
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

    nc = np.sum(splitting)
    n = A.shape[0]

    # Block BSR format. Transfer A to CSR and the splitting and SOC matrix to have
    # DOFs corresponding to CSR A
    if isspmatrix_bsr(A):
        temp_A = A.tocsr()
        splitting0 = splitting * np.ones((A.blocksize[0],1), dtype='intc')
        splitting0 = np.reshape(splitting0, (np.prod(splitting0.shape),), order='F')
        if theta is not None:
            C0 = classical_strength_of_connection(A, theta=theta, norm=norm, cost=cost)
            C0 = UnAmal(C0, A.blocksize[0], A.blocksize[1])
        else:
            C0 = UnAmal(C, A.blocksize[0], A.blocksize[1])
        C0 = C0.tocsr()

        # Use modified standard interpolation by ignoring strong F-connections that do
        # not have a common C-point.
        if modified:
            amg_core.remove_strong_FF_connections(temp_A.shape[0], C0.indptr, C0.indices,
                                                  C0.data, splitting)
        C0.eliminate_zeros()

        # Interpolation weights are computed based on entries in A, but subject to
        # the sparsity pattern of C.  So, copy the entries of A into the
        # sparsity pattern of C.
        C0.data[:] = 1.0
        C0 = C0.multiply(temp_A)

        P_indptr = np.empty_like(temp_A.indptr)
        amg_core.rs_standard_interpolation_pass1(temp_A.shape[0], C0.indptr,
                                                 C0.indices, splitting0, P_indptr)
        nnz = P_indptr[-1]
        P_colinds = np.empty(nnz, dtype=P_indptr.dtype)
        P_data = np.empty(nnz, dtype=temp_A.dtype)

        if modified:
            amg_core.mod_standard_interpolation_pass2(temp_A.shape[0], temp_A.indptr, temp_A.indices,
                                                      temp_A.data, C0.indptr, C0.indices,
                                                      C0.data, splitting0, P_indptr,
                                                      P_colinds, P_data)
        else:
            amg_core.rs_standard_interpolation_pass2(temp_A.shape[0], temp_A.indptr, temp_A.indices,
                                                     temp_A.data, C0.indptr, C0.indices,
                                                     C0.data, splitting0, P_indptr,
                                                     P_colinds, P_data)

        nc = np.sum(splitting0)
        n = A.shape[0] 
        P = csr_matrix((P_data, P_colinds, P_indptr), shape=[n,nc])
        return P.tobsr(blocksize=A.blocksize)

    # CSR format
    else:
        if theta is not None:
            C0 = classical_strength_of_connection(A, theta=theta, norm=norm, cost=cost)
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
        nc = np.sum(splitting)
        n = A.shape[0]
        return csr_matrix((P_data, P_colinds, P_indptr), shape=[n,nc])


def distance_two_interpolation(A, C, splitting, theta=None, norm='min', cost=[0]):
    #
    #
    # TODO: there is something wrong with the C-version of this
    #
    #
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
        theta value defining strong connections in a classical AMG sense. Provide if
        different SOC used for P than for CF-splitting; otherwise, theta = None. 
    norm : string, default 'abs'
        Norm used in redefining classical SOC. Options are 'min' and 'abs' for CSR matrices,
        and 'min', 'abs', and 'fro' for BSR matrices. See strength.py for more information.

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

    nc = np.sum(splitting)
    n = A.shape[0]

    # Block BSR format. Transfer A to CSR and the splitting and SOC matrix to have
    # DOFs corresponding to CSR A
    if isspmatrix_bsr(A):
        temp_A = A.tocsr()
        splitting0 = splitting * np.ones((A.blocksize[0],1), dtype='intc')
        splitting0 = np.reshape(splitting0, (np.prod(splitting0.shape),), order='F')
        if theta is not None:
            C0 = classical_strength_of_connection(A, theta=theta, norm=norm, cost=cost)
            C0 = UnAmal(C0, A.blocksize[0], A.blocksize[1])
        else:
            C0 = UnAmal(C, A.blocksize[0], A.blocksize[1])
        C0 = C0.tocsr()
        C0.eliminate_zeros()

        # Interpolation weights are computed based on entries in A, but subject to
        # the sparsity pattern of C.  So, copy the entries of A into the
        # sparsity pattern of C.
        C0.data[:] = 1.0
        C0 = C0.multiply(temp_A)

        P_indptr = np.empty_like(temp_A.indptr)
        amg_core.distance_two_amg_interpolation_pass1(temp_A.shape[0], C0.indptr,
                                                      C0.indices, splitting0, P_indptr)
        nnz = P_indptr[-1]
        P_colinds = np.empty(nnz, dtype=P_indptr.dtype)
        P_data = np.empty(nnz, dtype=temp_A.dtype)
        amg_core.distance_two_amg_interpolation_pass2(temp_A.shape[0], temp_A.indptr, temp_A.indices,
                                                      temp_A.data, C0.indptr, C0.indices,
                                                      C0.data, splitting0, P_indptr,
                                                      P_colinds, P_data)

        nc = np.sum(splitting0)
        n = A.shape[0] 
        P = csr_matrix((P_data, P_colinds, P_indptr), shape=[n,nc])
        return P.tobsr(blocksize=A.blocksize)

    # CSR format
    else:
        if theta is not None:
            C0 = classical_strength_of_connection(A, theta=theta, norm=norm, cost=cost)
        else:
            C0 = C.copy()
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
        amg_core.distance_two_amg_interpolation_pass2(A.shape[0], A.indptr, A.indices,
                                                      A.data, C0.indptr, C0.indices,
                                                      C0.data, splitting, P_indptr,
                                                      P_colinds, P_data)
        nc = np.sum(splitting)
        n = A.shape[0]
        return csr_matrix((P_data, P_colinds, P_indptr), shape=[n,nc])


def injection_interpolation(A, splitting, cost=[0]):
    """ Create interpolation operator by injection, that is C-points are
    interpolated by value and F-points are not interpolated.

    Parameters
    ----------
    A : {csr_matrix}
        NxN matrix in CSR format or BSR format
    splitting : array
        C/F splitting stored in an array of length N

    Returns
    -------
    NxNc interpolation operator, P
    """
    if isspmatrix_bsr(A):
        blocksize = A.blocksize[0]
        n = A.shape[0] / blocksize
    elif isspmatrix_csr(A):
        n = A.shape[0]
        blocksize = 1
    else:
        try:
            A = A.tocsr()
            warn("Implicit conversion of A to csr", SparseEfficiencyWarning)
            n = A.shape[0]
            blocksize = 1
        except:
            raise TypeError("Invalid matrix type, must be CSR or BSR.")

    P_rowptr = np.append(np.array([0],dtype='int32'), np.cumsum(splitting,dtype='int32') )
    nc = P_rowptr[-1]
    P_colinds = np.arange(start=0, stop=nc, step=1, dtype='int32')

    if blocksize == 1:
        return csr_matrix((np.ones((nc,), dtype=A.dtype), P_colinds, P_rowptr), shape=[n,nc])
    else:
        P_data = np.array(nc*[np.identity(blocksize, dtype=A.dtype)], dtype=A.dtype)
        return bsr_matrix((P_data, P_colinds, P_rowptr), blocksize=[blocksize,blocksize],
                          shape=[n*blocksize,nc*blocksize])


def one_point_interpolation(A, C, splitting, cost=[0]):
    """ Create one-point interpolation operator, that is C-points are
    interpolated by value and F-points are interpolated by value from
    their strongest-connected C-point neighbor.

    Parameters
    ----------
    A : {csr_matrix}
        NxN matrix in CSR format
    C : {csr_matrix}
        Strength-of-Connection matrix (does not need zero diagonal)
    splitting : array
        C/F splitting stored in an array of length N

    Returns
    -------
    NxNc interpolation operator, P
    """
    if isspmatrix_bsr(A):
        blocksize = A.blocksize[0]
        n = A.shape[0] / blocksize
    elif isspmatrix_csr(A):
        n = A.shape[0]
        blocksize = 1
    else:
        try:
            A = A.tocsr()
            warn("Implicit conversion of A to csr", SparseEfficiencyWarning)
            n = A.shape[0]
            blocksize = 1
        except:
            raise TypeError("Invalid matrix type, must be CSR or BSR.")

    nc = np.sum(splitting)
    P_rowptr = np.arange(start=0, stop=(n+1), step=1, dtype='int32')
    P_colinds = np.empty((n,),dtype='int32')
    amg_core.one_point_interpolation(P_rowptr, P_colinds, C.indptr,
                                     C.indices, C.data, splitting)
    if blocksize == 1:
        P_data = np.ones((n,), dtype=A.dtype)
        return csr_matrix((P_data,P_colinds,P_rowptr), shape=[n,nc])
    else:
        P_data = np.array(n*[np.identity(blocksize, dtype=A.dtype)], dtype=A.dtype)
        return bsr_matrix((P_data,P_colinds,P_rowptr), blocksize=[blocksize,blocksize],
                          shape=[blocksize*n,blocksize*nc])