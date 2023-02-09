"""Classical AMG Interpolation methods."""

from warnings import warn

import numpy as np
from scipy.sparse import csr_matrix, bsr_matrix, isspmatrix_csr, \
    isspmatrix_bsr, SparseEfficiencyWarning

from .. import amg_core
from ..strength import classical_strength_of_connection


def direct_interpolation(A, C, splitting, theta=None, norm='min'):
    """Create prolongator using direct interpolation.

    Parameters
    ----------
    A : csr_matrix
        NxN matrix in CSR format
    C : csr_matrix
        Strength-of-Connection matrix
        Must have zero diagonal
    theta : float in [0, 1), default None
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

    amg_core.rs_direct_interpolation_pass2(A.shape[0], A.indptr, A.indices, A.data,
                                           C.indptr, C.indices, C.data,
                                           splitting, P_indptr, P_indices, P_data)

    nc = np.sum(splitting)
    n = A.shape[0]
    return csr_matrix((P_data, P_indices, P_indptr), shape=[n, nc])


def classical_interpolation(A, C, splitting, theta=None, norm='min', modified=True):
    """Create prolongator using distance-1 classical interpolation.

    Parameters
    ----------
    A : csr_matrix
        NxN matrix in CSR format
    C : csr_matrix
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
    P : csr_matrix
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
        amg_core.remove_strong_FF_connections(A.shape[0], C.indptr, C.indices,
                                              C.data, splitting)
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

    return csr_matrix((P_data, P_indices, P_indptr), shape=[n, nc])


def injection_interpolation(A, splitting):
    """Create interpolation operator by injection.

    Parameters
    ----------
    A : {csr_matrix}
        NxN matrix in CSR format or BSR format
    splitting : array
        C/F splitting stored in an array of length N

    Returns
    -------
    NxNc interpolation operator, P

    Notes
    -----
    C-points are interpolated by value and F-points are not interpolated.

    Examples
    --------
    >>> from pyamg.gallery import poisson
    >>> from pyamg.classical.interpolate import injection_interpolation
    >>> import numpy as np
    >>> A = poisson((5,),format='csr')
    >>> splitting = np.array([1,0,1,0,1], dtype='intc')
    >>> P = injection_interpolation(A, splitting)
    >>> print(P.todense())
    [[1. 0. 0.]
     [0. 0. 0.]
     [0. 1. 0.]
     [0. 0. 0.]
     [0. 0. 1.]]
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
            warn('Implicit conversion of A to csr', SparseEfficiencyWarning)
            n = A.shape[0]
            blocksize = 1
        except BaseException as e:
            raise TypeError('Invalid matrix type, must be CSR or BSR.') from e

    P_rowptr = np.append(np.array([0], dtype=A.indptr.dtype),
                         np.cumsum(splitting, dtype=A.indptr.dtype))
    nc = P_rowptr[-1]
    P_colinds = np.arange(start=0, stop=nc, step=1, dtype=A.indptr.dtype)

    if blocksize == 1:
        P = csr_matrix((np.ones((nc,), dtype=A.dtype),
                       P_colinds, P_rowptr), shape=[n, nc])
    else:
        P_data = np.array(nc*[np.identity(blocksize, dtype=A.dtype)], dtype=A.dtype)
        P = bsr_matrix((P_data, P_colinds, P_rowptr), blocksize=[blocksize, blocksize],
                       shape=[n*blocksize, nc*blocksize])

    return P


def one_point_interpolation(A, C, splitting, by_val=False):
    """Create one-point interpolation operator.

    Parameters
    ----------
    A : {csr_matrix}
        NxN matrix in CSR format
    C : {csr_matrix}
        Strength-of-Connection matrix (does not need zero diagonal)
    splitting : array
        C/F splitting stored in an array of length N
    by_val : bool
        For CSR matrices only right now, use values of -Afc in interp as an
        approximation to P_ideal. If false, F-points are interpolated by value
        with weight 1.

    Returns
    -------
    NxNc interpolation operator, P

    Notes
    -----
    C-points are interpolated by value and F-points are interpolated by value
    from their strongest-connected C-point neighbor.

    Examples
    --------
    >>> from pyamg.gallery import poisson
    >>> from pyamg.classical.interpolate import one_point_interpolation
    >>> import numpy as np
    >>> A = poisson((5,),format='csr')
    >>> splitting = np.array([1,0,1,0,1], dtype='intc')
    >>> P = one_point_interpolation(A, A, splitting)
    >>> print(P.todense())
    [[1. 0. 0.]
     [1. 0. 0.]
     [0. 1. 0.]
     [0. 1. 0.]
     [0. 0. 1.]]
    """
    if isspmatrix_bsr(A):
        blocksize = A.blocksize[0]
        n = int(A.shape[0] / blocksize)
    elif isspmatrix_csr(A):
        n = A.shape[0]
        blocksize = 1
    else:
        try:
            A = A.tocsr()
            warn('Implicit conversion of A to csr', SparseEfficiencyWarning)
            n = A.shape[0]
            blocksize = 1
        except BaseException as e:
            raise TypeError('Invalid matrix type, must be CSR or BSR.') from e

    nc = np.sum(splitting)
    P_rowptr = np.empty((n+1,), dtype=A.indptr.dtype)  # P: n x nc, at most 'n' nnz
    P_colinds = np.empty((n,), dtype=A.indptr.dtype)
    P_data = np.empty((n,), dtype=A.dtype)

    if blocksize == 1:
        if by_val:
            amg_core.one_point_interpolation(P_rowptr, P_colinds, P_data, A.indptr,
                                             A.indices, A.data, splitting)
            P = csr_matrix((P_data, P_colinds, P_rowptr), shape=[n, nc])
        else:
            amg_core.one_point_interpolation(P_rowptr, P_colinds, P_data, C.indptr,
                                             C.indices, C.data, splitting)
            P_data = np.ones((n,), dtype=A.dtype)
            P = csr_matrix((P_data, P_colinds, P_rowptr), shape=[n, nc])
    else:
        amg_core.one_point_interpolation(P_rowptr, P_colinds, P_data, C.indptr,
                                         C.indices, C.data, splitting)
        P_data = np.array(n*[np.identity(blocksize, dtype=A.dtype)], dtype=A.dtype)
        P = bsr_matrix((P_data, P_colinds, P_rowptr), blocksize=[blocksize, blocksize],
                       shape=[blocksize*n, blocksize*nc])

    return P


def local_air(A, splitting, theta=0.1, norm='abs', degree=1,
              use_gmres=False, maxiter=10, precondition=True):
    """Compute approx ideal restriction by setting RA = 0, within sparsity pattern of R.

    Parameters
    ----------
    A : {csr_matrix, bsr_matrix}
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
        Solve local linear system for each row of R using GMRES.  If False, use
        direct solve.
    maxiter : int
        Maximum number of GMRES iterations
    precondition : bool
        Diagonally precondition GMRES

    Returns
    -------
    Approximate ideal restriction, R, in same sparse format as A.

    Notes
    -----
    Supports BSR (block) matrices, in addition to CSR.

    Sparsity pattern of R for the ith row (i.e. ith C-point) is the set of all
    strongly connected F-points, or the max_row *most* strongly connected
    F-points

    Examples
    --------
    >>> from pyamg.gallery import poisson
    >>> from pyamg.classical.interpolate import local_air
    >>> import numpy as np
    >>> A = poisson((5,),format='csr')
    >>> splitting = np.array([1,0,1,0,1], dtype='intc')
    >>> R = local_air(A, splitting)
    >>> print(R.todense())
    [[1.  0.5 0.  0.  0. ]
     [0.  0.5 1.  0.5 0. ]
     [0.  0.  0.  0.5 1. ]]
    """
    # Get SOC matrix containing neighborhood to be included in local solve
    if isspmatrix_bsr(A):
        C = classical_strength_of_connection(A=A, theta=theta, block=True, norm=norm)
        blocksize = A.blocksize[0]
    elif isspmatrix_csr(A):
        blocksize = 1
        C = classical_strength_of_connection(A=A, theta=theta, block=False, norm=norm)
    else:
        try:
            A = A.tocsr()
            warn('Implicit conversion of A to csr', SparseEfficiencyWarning)
            C = classical_strength_of_connection(A=A, theta=theta, block=False, norm=norm)
            blocksize = 1
        except BaseException as e:
            raise TypeError('Invalid matrix type, must be CSR or BSR.') from e

    Cpts = np.array(np.where(splitting == 1)[0], dtype=A.indptr.dtype)
    nc = Cpts.shape[0]

    R_rowptr = np.empty(nc+1, dtype=A.indptr.dtype)
    amg_core.approx_ideal_restriction_pass1(R_rowptr, C.indptr, C.indices,
                                            Cpts, splitting, degree)

    # Build restriction operator
    nnz = R_rowptr[-1]
    R_colinds = np.zeros(nnz, dtype=A.indptr.dtype)

    # Block matrix
    if isspmatrix_bsr(A):
        R_data = np.zeros(nnz*blocksize*blocksize, dtype=A.dtype)
        amg_core.block_approx_ideal_restriction_pass2(R_rowptr, R_colinds, R_data, A.indptr,
                                                      A.indices, A.data.ravel(), C.indptr,
                                                      C.indices, C.data, Cpts, splitting,
                                                      blocksize, degree, use_gmres, maxiter,
                                                      precondition)
        R = bsr_matrix((R_data.reshape((nnz, blocksize, blocksize)), R_colinds, R_rowptr),
                       blocksize=[blocksize, blocksize], shape=[nc*blocksize, A.shape[0]])
    # Not block matrix
    else:
        R_data = np.zeros(nnz, dtype=A.dtype)
        amg_core.approx_ideal_restriction_pass2(R_rowptr, R_colinds, R_data, A.indptr,
                                                A.indices, A.data, C.indptr, C.indices,
                                                C.data, Cpts, splitting, degree, use_gmres,
                                                maxiter, precondition)
        R = csr_matrix((R_data, R_colinds, R_rowptr), shape=[nc, A.shape[0]])

    R.eliminate_zeros()
    return R
