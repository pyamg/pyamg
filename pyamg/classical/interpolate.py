"""Classical AMG Interpolation methods."""
import numpy as np
from scipy.sparse import csr_matrix, isspmatrix_csr
from pyamg import amg_core
from pyamg.util.utils import filter_matrix_rows, UnAmal
from pyamg.strength import classical_strength_of_connection


__all__ = ['direct_interpolation','standard_interpolation',
           'distance_two_interpolation','injection_interpolation',
           'one_point_interpolation','neumann_AIR', 'local_AIR']


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
    
    if theta is not None:
        C0 = classical_strength_of_connection(A, theta=theta, norm=norm, cost=cost)
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
        C0 = classical_strength_of_connection(A, theta=theta, norm=norm, cost=cost)
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
        C0 = classical_strength_of_connection(A, theta=theta, norm=norm, cost=cost)
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


def injection_interpolation(A, splitting):
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


def one_point_interpolation(A, C, splitting, by_val=False):
    """ Create one-point interpolation operator, that is C-points are
    interpolated by value and F-points are interpolated by value from
    their strongest-connected C-point neighbor.

    Parameters
    ----------
    A : {csr_matrix}
        NxN matrix in CSR format
    C : {csr_matrix}
        Strength-of-Connection matrix (does not need zero diagonal)
    by_val : bool
        For CSR matrices only right now, use values of -Afc in interp as an
        approximation to P_ideal. If false, F-points are interpolated by value
        with weight 1.
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
    P_rowptr = np.empty((n+1,), dtype='int32') # P: n x nc, at most 'n' nnz
    P_colinds = np.empty((n,),dtype='int32')
    P_data = np.empty((n,),dtype=A.dtype)

    #amg_core.one_point_interpolation(P_rowptr, P_colinds, A.indptr,
    #                                 A.indices, A.data, splitting)
    if blocksize == 1:
        if by_val:
            amg_core.one_point_interpolation(P_rowptr, P_colinds, P_data, A.indptr,
                                     A.indices, A.data, splitting)
            return csr_matrix((P_data,P_colinds,P_rowptr), shape=[n,nc])
        else:
            amg_core.one_point_interpolation(P_rowptr, P_colinds, P_data, C.indptr,
                                     C.indices, C.data, splitting)
            P_data = np.ones((n,), dtype=A.dtype)
            return csr_matrix((P_data,P_colinds,P_rowptr), shape=[n,nc])
    else:
        amg_core.one_point_interpolation(P_rowptr, P_colinds, P_data, C.indptr,
                         C.indices, C.data, splitting)
        P_data = np.array(n*[np.identity(blocksize, dtype=A.dtype)], dtype=A.dtype)
        return bsr_matrix((P_data,P_colinds,P_rowptr), blocksize=[blocksize,blocksize],
                          shape=[blocksize*n,blocksize*nc])


def neumann_AIR(A, splitting, theta=0.025, degree=1, post_theta=0):
    """ Approximate ideal restriction using a truncated Neumann expansion for A_ff^{-1},
    where 
        R = [-Acf*D, I],   where
        D = \sum_{i=0}^degree Lff^i

    Parameters
    ----------
    A : {csr_matrix}
        NxN matrix in CSR format
    splitting : array
        C/F splitting stored in an array of length N
    theta : float : default 0.025
        Compute approximation to ideal restriction for C, where C has rows filtered
        with tolerance theta, that is for j s.t.
            |C_ij| <= theta * |C_ii|        --> C_ij = 0.
        Helps keep R sparse. 
    degree : int in [0,4] : default 1
        Degree of Neumann expansion. Only supported up to degree 4.

    Returns
    -------
    Approximate ideal restriction in CSR format.

    Notes
    -----
    Does not support block matrices.
    """

    Cpts = np.array(np.where(splitting == 1)[0], dtype='int32')
    Fpts = np.array(np.where(splitting == 0)[0], dtype='int32')

    # Convert block CF-splitting into scalar CF-splitting so that we can access
    # submatrices of BSR matrix A
    if isspmatrix_bsr(A):
        bsize = A.blocksize[0]
        Cpts *= bsize
        Fpts *= bsize
        Cpts0 = Cpts
        Fpts0 = Fpts
        for i in range(1,bsize):
            Cpts = np.hstack([Cpts,Cpts0+i])
            Fpts = np.hstack([Fpts,Fpts0+i])
        Cpts.sort()
        Fpts.sort()
    
    nc = Cpts.shape[0]
    nf = Fpts.shape[0]
    n = A.shape[0]
    C = csr_matrix(A, copy=True)
    if theta > 0.0:
        filter_matrix_rows(C, theta, diagonal=True, lump=False)

    # Expand sparsity pattern for R
    C.data[np.abs(C.data)<1e-16] = 0
    C.eliminate_zeros()

    Lff = -C[Fpts,:][:,Fpts]
    pts = np.arange(0,nf)
    Lff[pts,pts] = 0.0
    Lff.eliminate_zeros()
    Acf = C[Cpts,:][:,Fpts]

    # Form Neuman approximation to Aff^{-1}
    Z = eye(nf,format='csr')
    if degree >= 1:
        Z += Lff
    if degree >= 2:
        Z += Lff*Lff
    if degree >= 3:
        Z += Lff*Lff*Lff
    if degree == 4:
        Z += Lff*Lff*Lff*Lff
    if degree > 4:
        raise ValueError("Only sparsity degree 0-4 supported.")

    # Multiply Acf by approximation to Aff^{-1}
    Z = -Acf*Z

    if post_theta > 0.0:
        if not isspmatrix_csr(Z):
            Z = Z.tocsr()
        filter_matrix_rows(Z, post_theta, diagonal=False, lump=False)

    # Get sizes and permutation matrix from [F, C] block
    # ordering to natural matrix ordering.
    permute = eye(n,format='csr')
    permute.indices = np.concatenate((Fpts,Cpts))

    # Form R = [Z, I], reorder and return
    R = hstack([Z, eye(nc, format='csr')])
    if isspmatrix_bsr(A):
        R = bsr_matrix(R * permute, blocksize=[bsize,bsize])
    else:
        R = csr_matrix(R * permute)
    return R


def local_AIR(A, splitting, theta=0.1, norm='abs', degree=1, use_gmres=False,
                                  maxiter=10, precondition=True):
    """ Compute approximate ideal restriction by setting RA = 0, within the
    sparsity pattern of R. Sparsity pattern of R for the ith row (i.e. ith
    C-point) is the set of all strongly connected F-points, or the max_row
    *most* strongly connected F-points.

    Parameters
    ----------
    A : {csr_matrix}
        NxN matrix in CSR or BSR format
    splitting : array
        C/F splitting stored in an array of length N
    theta : float, default 0.1
        Solve local system for each row of R for all values
            |A_ij| >= 0.1 * max_{i!=k} |A_ik|
    degree : int, default 1
        Expand sparsity pattern for R by considering strongly connected
        neighbors within 'degree' of a given node. Only supports degree 1 and 2.
    use_gmres : bool
        Solve local linear system for each row of R using GMRES
    maxiter : int
        Maximum number of GMRES iterations
    precondition : bool
        Diagonally precondition GMRES

    Returns
    -------
    Approximate ideal restriction, R, in same sparse format as A.

    Notes
    -----
    - This was the original idea for approximating ideal restriction. In practice,
      however, a Neumann approximation is typically used.
    - Supports block bsr matrices as well.
    """

    # Get SOC matrix containing neighborhood to be included in local solve
    if isspmatrix_bsr(A):
        C = classical_strength_of_connection(A=A, theta=theta, block='amalgamate', norm=norm)
        blocksize = A.blocksize[0]
    elif isspmatrix_csr(A):
        blocksize = 1
        C = classical_strength_of_connection(A=A, theta=theta, block=None, norm=norm)
    else:
        try:
            A = A.tocsr()
            warn("Implicit conversion of A to csr", SparseEfficiencyWarning)
            C = classical_strength_of_connection(A=A, theta=theta, block=None, norm=norm)
            blocksize = 1
        except:
            raise TypeError("Invalid matrix type, must be CSR or BSR.")

    Cpts = np.array(np.where(splitting == 1)[0], dtype='int32')
    nc = Cpts.shape[0]
    n = C.shape[0]

    R_rowptr = np.empty(nc+1, dtype='int32')
    amg_core.approx_ideal_restriction_pass1(R_rowptr, C.indptr, C.indices,
                                            Cpts, splitting, degree)       

    # Build restriction operator
    nnz = R_rowptr[-1]
    R_colinds = np.zeros(nnz, dtype='int32')

    # Block matrix
    if isspmatrix_bsr(A):
        R_data = np.zeros(nnz*blocksize*blocksize, dtype=A.dtype)
        amg_core.block_approx_ideal_restriction_pass2(R_rowptr, R_colinds, R_data, A.indptr,
                                                      A.indices, A.data.ravel(), C.indptr,
                                                      C.indices, C.data, Cpts, splitting,
                                                      blocksize, degree, use_gmres, maxiter,
                                                      precondition)
        R = bsr_matrix((R_data.reshape(nnz,blocksize,blocksize), R_colinds, R_rowptr),
                        blocksize=[blocksize,blocksize], shape=[nc*blocksize,A.shape[0]])
    # Not block matrix
    else:
        R_data = np.zeros(nnz, dtype=A.dtype)
        amg_core.approx_ideal_restriction_pass2(R_rowptr, R_colinds, R_data, A.indptr,
                                                A.indices, A.data, C.indptr, C.indices,
                                                C.data, Cpts, splitting, degree, use_gmres, maxiter,
                                                precondition)            
        R = csr_matrix((R_data, R_colinds, R_rowptr), shape=[nc,A.shape[0]])

    R.eliminate_zeros()
    return R
