"""Methods to smooth tentative prolongation operators."""


from warnings import warn
import numpy as np
from scipy import sparse
import scipy.linalg as la
from scipy.linalg import lapack as lapack
from .. import amg_core
from ..util.utils import scale_rows, get_diagonal, get_block_diag, \
    unamal, filter_operator, compute_BtBinv, filter_matrix_rows, \
    truncate_rows
from ..util.linalg import approximate_spectral_radius
from ..util import upcast
from ..util.params import set_tol
from .. import amg_core
from ..strength import classical_strength_of_connection

# satisfy_constraints is a helper function for prolongation smoothing routines
def satisfy_constraints(U, B, BtBinv):
    """U is the prolongator update.  Project out components of U such that U@B = 0.

    Parameters
    ----------
    U : bsr_array
        m x n sparse bsr matrix
        Update to the prolongator
    B : array
        n x k array of the coarse grid near nullspace vectors
    BtBinv : array
        Local inv(B_i.H@B_i) matrices for each supernode, i
        B_i is B restricted to the sparsity pattern of supernode i in U

    Returns
    -------
    Updated U, so that U@B = 0.
    Update is computed by orthogonally (in 2-norm) projecting
    out the components of span(B) in U in a row-wise fashion.

    See Also
    --------
    The principal calling routine,
    pyamg.aggregation.smooth.energy_prolongation_smoother

    """
    rows_per_block = U.blocksize[0]
    cols_per_block = U.blocksize[1]
    num_block_rows = int(U.shape[0]/rows_per_block)

    UB = np.ravel(U@B)

    # Apply constraints, noting that we need the conjugate of B
    # for use as Bi.H in local projection
    amg_core.satisfy_constraints_helper(rows_per_block, cols_per_block,
                                        num_block_rows, B.shape[1],
                                        np.conjugate(np.ravel(B)),
                                        UB, np.ravel(BtBinv),
                                        U.indptr, U.indices,
                                        np.ravel(U.data))

    return U


def jacobi_prolongation_smoother(S, T, C, B, omega=4.0/3.0, degree=1,
                                 filter_entries=False, weighting='diagonal'):
    """Jacobi prolongation smoother.

    Parameters
    ----------
    S : csr_array, bsr_array
        Sparse NxN matrix used for smoothing.  Typically, A.
    T : csr_array, bsr_array
        Tentative prolongator.
    C : csr_array, bsr_array
        Strength-of-connection matrix.
    B : array
        Near nullspace modes for the coarse grid such that T@B
        exactly reproduces the fine grid near nullspace modes.
    omega : scalar
        Damping parameter.
    degree : int
        Number of passes.
    filter_entries : bool
        If true, filter S before smoothing T.  This option can greatly control
        complexity.
    weighting : str
        'block', 'diagonal' or 'local' weighting for constructing the Jacobi D
        'local' Uses a local row-wise weight based on the Gershgorin estimate.
        Avoids any potential under-damping due to inaccurate spectral radius
        estimates.
        'block' uses a block diagonal inverse of A if A is BSR
        'diagonal' uses classic Jacobi with D = diagonal(A).

    Returns
    -------
    csr_array, bsr_array
        Smoothed (final) prolongator defined by P = (I - omega/rho(K) K) @ T
        where K = diag(S)^-1 @ S and rho(K) is an approximation to the
        spectral radius of K.

    Notes
    -----
    If weighting is not 'local', then results using Jacobi prolongation
    smoother are not precisely reproducible due to a random initial guess used
    for the spectral radius approximation.  For precise reproducibility,
    set numpy.random.seed(..) to the same value before each test.

    Examples
    --------
    >>> from pyamg.aggregation import jacobi_prolongation_smoother
    >>> from pyamg.gallery import poisson
    >>> from scipy.sparse import coo_array
    >>> import numpy as np
    >>> data = np.ones((6,))
    >>> row = np.arange(0,6)
    >>> col = np.kron([0, 1], np.ones(3, dtype=int))
    >>> T = coo_array((data,(row, col)),shape=(6, 2)).tocsr()
    >>> T.toarray()
    array([[1., 0.],
           [1., 0.],
           [1., 0.],
           [0., 1.],
           [0., 1.],
           [0., 1.]])
    >>> A = poisson((6,),format='csr')
    >>> P = jacobi_prolongation_smoother(A,T,A,np.ones((2,1)))
    >>> P.toarray()
    array([[0.64930164, 0.        ],
           [1.        , 0.        ],
           [0.64930164, 0.35069836],
           [0.35069836, 0.64930164],
           [0.        , 1.        ],
           [0.        , 0.64930164]])

    """
    # preprocess weighting
    if weighting == 'block':
        if sparse.issparse(S) and S.format == 'csr':
            weighting = 'diagonal'
        elif sparse.issparse(S) and S.format == 'bsr':
            if S.blocksize[0] == 1:
                weighting = 'diagonal'
        else:
            raise TypeError('S must be sparse BSR or CSR format')

    if filter_entries:
        # Implement filtered prolongation smoothing for the general case by
        # utilizing satisfy constraints

        if sparse.issparse(S) and S.format == 'bsr':
            numPDEs = S.blocksize[0]
        else:
            numPDEs = 1

        # Create a filtered S with entries dropped that aren't in C
        C = unamal(C, numPDEs, numPDEs)
        S = S.multiply(C)
        S.eliminate_zeros()

    if weighting == 'diagonal':
        # Use diagonal of S
        D_inv = get_diagonal(S, inv=True)
        D_inv_S = scale_rows(S, D_inv, copy=True)
        D_inv_S = (omega/approximate_spectral_radius(D_inv_S))*D_inv_S
    elif weighting == 'block':
        # Use block diagonal of S
        D_inv = get_block_diag(S, blocksize=S.blocksize[0], inv_flag=True)
        D_inv = sparse.bsr_array((D_inv, np.arange(D_inv.shape[0], dtype=np.int32),
                                   np.arange(D_inv.shape[0] + 1, dtype=np.int32)),
                                  shape=S.shape)
        D_inv_S = D_inv@S
        D_inv_S = (omega/approximate_spectral_radius(D_inv_S))*D_inv_S
    elif weighting == 'local':
        # Use the Gershgorin estimate as each row's weight, instead of a global
        # spectral radius estimate
        D = np.abs(S)@np.ones((S.shape[0], 1), dtype=S.dtype)
        D_inv = np.zeros_like(D)
        D_inv[D != 0] = 1.0 / np.abs(D[D != 0])

        D_inv_S = scale_rows(S, D_inv, copy=True)
        D_inv_S = omega*D_inv_S
    else:
        raise ValueError('Incorrect weighting option')

    if filter_entries:
        # Carry out Jacobi, but after calculating the prolongator update, U,
        # apply satisfy constraints so that U@B = 0
        P = T
        for _ in range(degree):
            U = (D_inv_S@P).tobsr(blocksize=P.blocksize)

            # Enforce U@B = 0 (1) Construct array of inv(Bi'Bi), where Bi is B
            # restricted to row i's sparsity pattern in pattern. This
            # array is used multiple times in satisfy_constraints(...).
            BtBinv = compute_BtBinv(B, U)

            # (2) Apply satisfy constraints
            satisfy_constraints(U, B, BtBinv)

            # Update P
            P = P - U

    else:
        # Carry out Jacobi as normal
        P = T
        for _ in range(degree):
            P = P - (D_inv_S @ P)

    return P


def richardson_prolongation_smoother(S, T, omega=4.0/3.0, degree=1):
    """Richardson prolongation smoother.

    Parameters
    ----------
    S : csr_array, bsr_array
        Sparse NxN matrix used for smoothing.  Typically, A or the
        "filtered matrix" obtained from A by lumping weak connections
        onto the diagonal of A.
    T : csr_array, bsr_array
        Tentative prolongator.
    omega : scalar
        Damping parameter.
    degree : int
        Number of passes.

    Returns
    -------
    csr_array, bsr_array
        Smoothed (final) prolongator defined by P = (I - omega/rho(S) S) @ T
        where rho(S) is an approximation to the spectral radius of S.

    Notes
    -----
    Results using Richardson prolongation smoother are not precisely
    reproducible due to a random initial guess used for the spectral radius
    approximation.  For precise reproducibility, set numpy.random.seed(..) to
    the same value before each test.

    Examples
    --------
    >>> from pyamg.aggregation import richardson_prolongation_smoother
    >>> from pyamg.gallery import poisson
    >>> from scipy.sparse import coo_array
    >>> import numpy as np
    >>> data = np.ones((6,))
    >>> row = np.arange(0, 6, dtype=np.int32)
    >>> col = np.kron([0, 1], np.ones(3, dtype=int))
    >>> T = coo_array((data,(row, col)),shape=(6, 2)).tocsr()
    >>> T.toarray()
    array([[1., 0.],
           [1., 0.],
           [1., 0.],
           [0., 1.],
           [0., 1.],
           [0., 1.]])
    >>> A = poisson((6,),format='csr')
    >>> P = richardson_prolongation_smoother(A,T)
    >>> P.toarray()
    array([[0.64930164, 0.        ],
           [1.        , 0.        ],
           [0.64930164, 0.35069836],
           [0.35069836, 0.64930164],
           [0.        , 1.        ],
           [0.        , 0.64930164]])

    """
    weight = omega/approximate_spectral_radius(S)

    P = T
    for _ in range(degree):
        P = P - weight*(S@P)

    return P


def cg_prolongation_smoothing(A, T, B, BtBinv, pattern, maxiter, tol,
                              weighting='local', Cpt_params=None):
    """Use CG to smooth T by solving A T = 0, subject to nullspace and sparsity constraints.

    Parameters
    ----------
    A : csr_array, bsr_array
        SPD sparse NxN matrix
    T : bsr_array
        Tentative prolongator, a NxM sparse matrix (M < N).
        This is initial guess for the equation A T = 0.
        Assumed that T B_c = B_f
    B : array
        Near-nullspace modes for coarse grid, i.e., B_c.
        Has shape (M,k) where k is the number of coarse candidate vectors.
    BtBinv : array
        3 dimensional array such that,
        BtBinv[i] = pinv(B_i.H Bi), and B_i is B restricted
        to the neighborhood (in the matrix graph) of dof of i.
    pattern : csr_array, bsr_array
        Sparse NxM matrix
        This is the sparsity pattern constraint to enforce on the
        eventual prolongator
    maxiter : int
        maximum number of iterations
    tol : float
        residual tolerance for A T = 0
    weighting : string
        'block', 'diagonal' or 'local' construction of the diagonal
        preconditioning
    Cpt_params : tuple
        Tuple of the form (bool, dict).  If the Cpt_params[0] = False, then
        the standard SA prolongation smoothing is carried out.  If True, then
        dict must be a dictionary of parameters containing, (1) P_I: P_I.T is
        the injection matrix for the Cpts, (2) I_F: an identity matrix
        for only the F-points (i.e. I, but with zero rows and columns for
        C-points) and I_C: the C-point analogue to I_F.

    Returns
    -------
    T : bsr_array
        Smoothed prolongator using conjugate gradients to solve A T = 0,
        subject to the constraints, T B_c = B_f, and T has no nonzero
        outside of the sparsity pattern in pattern.

    See Also
    --------
    The principal calling routine,
    pyamg.aggregation.smooth.energy_prolongation_smoother

    """
    # Preallocate
    AP = sparse.bsr_array((np.zeros(pattern.data.shape, dtype=T.dtype),
                            pattern.indices, pattern.indptr),
                           shape=pattern.shape)

    # CG will be run with diagonal preconditioning
    if weighting == 'diagonal':
        Dinv = get_diagonal(A, norm_eq=False, inv=True)
    elif weighting == 'block':
        Dinv = get_block_diag(A, blocksize=A.blocksize[0], inv_flag=True)
        Dinv = sparse.bsr_array((Dinv, np.arange(Dinv.shape[0], dtype=np.int32),
                                 np.arange(Dinv.shape[0] + 1, dtype=np.int32)),
                                shape=A.shape)
    elif weighting == 'local':
        # Based on Gershgorin estimate
        D = np.abs(A)@np.ones((A.shape[0], 1), dtype=A.dtype)
        Dinv = np.zeros_like(D)
        Dinv[D != 0] = 1.0 / np.abs(D[D != 0])
    else:
        raise ValueError('weighting value is invalid')

    # Calculate initial residual
    #   Equivalent to R = -A@T;    R = R.multiply(pattern)
    #   with the added constraint that R has an explicit 0 wherever
    #   R is 0 and pattern is not
    uones = np.zeros(pattern.data.shape, dtype=T.dtype)
    R = sparse.bsr_array((uones, pattern.indices,
                           pattern.indptr),
                          shape=pattern.shape)
    amg_core.incomplete_mat_mult_bsr(A.indptr, A.indices,
                                     np.ravel(A.data),
                                     T.indptr, T.indices,
                                     np.ravel(T.data),
                                     R.indptr, R.indices,
                                     np.ravel(R.data),
                                     int(T.shape[0]/T.blocksize[0]),
                                     int(T.shape[1]/T.blocksize[1]),
                                     A.blocksize[0], A.blocksize[1],
                                     T.blocksize[1])
    R.data *= -1.0

    # Enforce R@B = 0
    satisfy_constraints(R, B, BtBinv)

    if R.nnz == 0:
        print('Error in sa_energy_min(..).  Initial R no nonzeros on a level. '
              'Returning tentative prolongator\n')
        return T

    # Calculate Frobenius norm of the residual
    resid = R.nnz  # np.sqrt((R.data.conjugate()*R.data).sum())
    # print "Energy Minimization of Prolongator \
    #       --- Iteration 0 --- r = " + str(resid)

    i = 0
    while i < maxiter and resid > tol:
        # Apply diagonal preconditioner
        if weighting in ('local', 'diagonal'):
            Z = scale_rows(R, Dinv)
        else:
            Z = Dinv@R

        # Frobenius inner-product of (R,Z) = sum( np.conjugate(rk).*zk)
        newsum = (R.conjugate().multiply(Z)).sum()
        if newsum < tol:
            # met tolerance, so halt
            break

        # P is the search direction, not the prolongator, which is T.
        if i == 0:
            P = Z
            oldsum = newsum
        else:
            beta = newsum / oldsum
            P = Z + beta*P
        oldsum = newsum

        # Calculate new direction and enforce constraints
        #   Equivalent to:  AP = A@P;    AP = AP.multiply(pattern)
        #   with the added constraint that explicit zeros are in AP wherever
        #   AP = 0 and pattern does not  !!!!
        AP.data[:] = 0.0
        amg_core.incomplete_mat_mult_bsr(A.indptr, A.indices,
                                         np.ravel(A.data),
                                         P.indptr, P.indices,
                                         np.ravel(P.data),
                                         AP.indptr, AP.indices,
                                         np.ravel(AP.data),
                                         int(T.shape[0]/T.blocksize[0]),
                                         int(T.shape[1]/T.blocksize[1]),
                                         A.blocksize[0], A.blocksize[1],
                                         P.blocksize[1])

        # Enforce AP@B = 0
        satisfy_constraints(AP, B, BtBinv)

        # Frobenius inner-product of (P, AP)
        alpha = newsum/(P.conjugate().multiply(AP)).sum()

        # Update the prolongator, T
        T = T + alpha*P

        # Ensure identity at C-pts
        if Cpt_params[0]:
            T = Cpt_params[1]['I_F']@T + Cpt_params[1]['P_I']

        # Update residual
        R = R - alpha*AP

        i += 1

        # Calculate Frobenius norm of the residual
        resid = R.nnz  # np.sqrt((R.data.conjugate()*R.data).sum())
        # print "Energy Minimization of Prolongator \
        # --- Iteration " + str(i) + " --- r = " + str(resid)

    return T


def cgnr_prolongation_smoothing(A, T, B, BtBinv, pattern, maxiter,
                                tol, weighting='diagonal', Cpt_params=None):
    """Smooth T with CGNR by solving A T = 0, subject to nullspace and sparsity constraints.

    Parameters
    ----------
    A : csr_array, bsr_array
        SPD sparse NxN matrix
        Should be at least nonsymmetric or indefinite
    T : bsr_array
        Tentative prolongator, a NxM sparse matrix (M < N).
        This is initial guess for the equation A T = 0.
        Assumed that T B_c = B_f
    B : array
        Near-nullspace modes for coarse grid, i.e., B_c.
        Has shape (M,k) where k is the number of coarse candidate vectors.
    BtBinv : array
        3 dimensional array such that,
        BtBinv[i] = pinv(B_i.H Bi), and B_i is B restricted
        to the neighborhood (in the matrix graph) of dof of i.
    pattern : csr_array, bsr_array
        Sparse NxM matrix
        This is the sparsity pattern constraint to enforce on the
        eventual prolongator
    maxiter : int
        maximum number of iterations
    tol : float
        residual tolerance for A T = 0
    weighting : string
        'block', 'diagonal' or 'local' construction of the diagonal
        preconditioning
        IGNORED here, only 'diagonal' preconditioning is used.
    Cpt_params : tuple
        Tuple of the form (bool, dict).  If the Cpt_params[0] = False, then
        the standard SA prolongation smoothing is carried out.  If True, then
        dict must be a dictionary of parameters containing, (1) P_I: P_I.T is
        the injection matrix for the Cpts, (2) I_F: an identity matrix
        for only the F-points (i.e. I, but with zero rows and columns for
        C-points) and I_C: the C-point analogue to I_F.

    Returns
    -------
    T : bsr_array
        Smoothed prolongator using CGNR to solve A T = 0,
        subject to the constraints, T B_c = B_f, and T has no nonzero
        outside of the sparsity pattern in pattern.

    See Also
    --------
    The principal calling routine,
    pyamg.aggregation.smooth.energy_prolongation_smoother

    """
    if weighting != 'diagonal':
        warn(f'Weighting of {weighting} unused.', stacklevel=2)

    # For non-SPD system, apply CG on Normal Equations with Diagonal
    # Preconditioning (requires transpose)
    Ah = A.T.conjugate()
    Ah.sort_indices()

    # Preallocate
    uones = np.zeros(pattern.data.shape, dtype=T.dtype)
    AP = sparse.bsr_array((uones, pattern.indices, pattern.indptr),
                           shape=pattern.shape)

    # D for A.H@A
    Dinv = get_diagonal(A, norm_eq=1, inv=True)

    # Calculate initial residual
    #   Equivalent to R = -Ah@(A@T);    R = R.multiply(pattern)
    #   with the added constraint that R has an explicit 0 wherever
    #   R is 0 and pattern is not
    uones = np.zeros(pattern.data.shape, dtype=T.dtype)
    R = sparse.bsr_array((uones, pattern.indices, pattern.indptr),
                          shape=pattern.shape)
    AT = -1.0*A@T
    R.data[:] = 0.0
    amg_core.incomplete_mat_mult_bsr(Ah.indptr, Ah.indices,
                                     np.ravel(Ah.data),
                                     AT.indptr, AT.indices,
                                     np.ravel(AT.data),
                                     R.indptr, R.indices,
                                     np.ravel(R.data),
                                     int(T.shape[0]/T.blocksize[0]),
                                     int(T.shape[1]/T.blocksize[1]),
                                     Ah.blocksize[0], Ah.blocksize[1],
                                     T.blocksize[1])

    # Enforce R@B = 0
    satisfy_constraints(R, B, BtBinv)

    if R.nnz == 0:
        print('Error in sa_energy_min(..).  Initial R no nonzeros on a level. '
              'Returning tentative prolongator')
        return T

    # Calculate Frobenius norm of the residual
    resid = R.nnz  # np.sqrt((R.data.conjugate()*R.data).sum())
    # print "Energy Minimization of Prolongator \
    # --- Iteration 0 --- r = " + str(resid)

    i = 0
    while i < maxiter and resid > tol:

        # vect = np.ravel((A@T).data)
        # print "Iteration " + str(i) + "   \
        # Energy = %1.3e"%np.sqrt( (vect.conjugate()*vect).sum() )

        # Apply diagonal preconditioner
        Z = scale_rows(R, Dinv)

        # Frobenius innerproduct of (R,Z) = sum(rk.*zk)
        newsum = (R.conjugate().multiply(Z)).sum()
        if newsum < tol:
            # met tolerance, so halt
            break

        # P is the search direction, not the prolongator, which is T.
        if i == 0:
            P = Z
            oldsum = newsum
        else:
            beta = newsum/oldsum
            P = Z + beta*P
        oldsum = newsum

        # Calculate new direction
        #  Equivalent to:  AP = Ah@(A@P);    AP = AP.multiply(pattern)
        #  with the added constraint that explicit zeros are in AP wherever
        #  AP = 0 and pattern does not
        AP_temp = A@P
        AP.data[:] = 0.0
        amg_core.incomplete_mat_mult_bsr(Ah.indptr, Ah.indices,
                                         np.ravel(Ah.data),
                                         AP_temp.indptr, AP_temp.indices,
                                         np.ravel(AP_temp.data),
                                         AP.indptr, AP.indices,
                                         np.ravel(AP.data),
                                         int(T.shape[0]/T.blocksize[0]),
                                         int(T.shape[1]/T.blocksize[1]),
                                         Ah.blocksize[0],
                                         Ah.blocksize[1], T.blocksize[1])
        del AP_temp

        # Enforce AP@B = 0
        satisfy_constraints(AP, B, BtBinv)

        # Frobenius inner-product of (P, AP)
        alpha = newsum/(P.conjugate().multiply(AP)).sum()

        # Update the prolongator, T
        T = T + alpha*P

        # Ensure identity at C-pts
        if Cpt_params[0]:
            T = Cpt_params[1]['I_F']@T + Cpt_params[1]['P_I']

        # Update residual
        R = R - alpha*AP

        i += 1

        # Calculate Frobenius norm of the residual
        resid = R.nnz  # np.sqrt((R.data.conjugate()*R.data).sum())
        # print "Energy Minimization of Prolongator \
        # --- Iteration " + str(i) + " --- r = " + str(resid)

    # vect = np.ravel((A@T).data)
    # print "Final Iteration " + str(i) + "   \
    # Energy = %1.3e"%np.sqrt( (vect.conjugate()*vect).sum() )

    return T


def apply_givens(Q, v, k):
    """Apply the first k Givens rotations in Q to v.

    Parameters
    ----------
    Q : list
        list of consecutive 2x2 Givens rotations
    v : array
        vector to apply the rotations to
    k : int
        number of rotations to apply.

    Returns
    -------
    v is changed in place

    Notes
    -----
    This routine is specialized for GMRES.  It assumes that the first Givens
    rotation is for dofs 0 and 1, the second Givens rotation is for dofs 1 and
    2, and so on.

    """
    for j in range(k):
        Qloc = Q[j]
        v[j:j+2] = np.dot(Qloc, v[j:j+2])


def gmres_prolongation_smoothing(A, T, B, BtBinv, pattern, maxiter,
                                 tol, weighting='local', Cpt_params=None):
    """Smooth T with GMRES by solving A T = 0 subject to nullspace and sparsity constraints.

    Parameters
    ----------
    A : csr_array, bsr_array
        SPD sparse NxN matrix
        Should be at least nonsymmetric or indefinite
    T : bsr_array
        Tentative prolongator, a NxM sparse matrix (M < N).
        This is initial guess for the equation A T = 0.
        Assumed that T B_c = B_f
    B : array
        Near-nullspace modes for coarse grid, i.e., B_c.
        Has shape (M,k) where k is the number of coarse candidate vectors.
    BtBinv : array
        3 dimensional array such that,
        BtBinv[i] = pinv(B_i.H Bi), and B_i is B restricted
        to the neighborhood (in the matrix graph) of dof of i.
    pattern : csr_array, bsr_array
        Sparse NxM matrix
        This is the sparsity pattern constraint to enforce on the
        eventual prolongator
    maxiter : int
        maximum number of iterations
    tol : float
        residual tolerance for A T = 0
    weighting : string
        'block', 'diagonal' or 'local' construction of the diagonal
        preconditioning
    Cpt_params : tuple
        Tuple of the form (bool, dict).  If the Cpt_params[0] = False, then
        the standard SA prolongation smoothing is carried out.  If True, then
        dict must be a dictionary of parameters containing, (1) P_I: P_I.T is
        the injection matrix for the Cpts, (2) I_F: an identity matrix
        for only the F-points (i.e. I, but with zero rows and columns for
        C-points) and I_C: the C-point analogue to I_F.

    Returns
    -------
    T : bsr_array
        Smoothed prolongator using GMRES to solve A T = 0,
        subject to the constraints, T B_c = B_f, and T has no nonzero
        outside of the sparsity pattern in pattern.

    See Also
    --------
    The principal calling routine,
    pyamg.aggregation.smooth.energy_prolongation_smoother

    """
    # For non-SPD system, apply GMRES with Diagonal Preconditioning

    # Preallocate space for new search directions
    uones = np.zeros(pattern.data.shape, dtype=T.dtype)
    AV = sparse.bsr_array((uones, pattern.indices, pattern.indptr),
                           shape=pattern.shape)

    # Preallocate for Givens Rotations, Hessenberg matrix and Krylov Space
    xtype = upcast(A.dtype, T.dtype, B.dtype)
    Q = []      # Givens Rotations
    V = []      # Krylov Space
    # vs = []     # vs store the pointers to each column of V for speed

    # Upper Hessenberg matrix, converted to upper tri with Givens Rots
    H = np.zeros((maxiter+1, maxiter+1), dtype=xtype)

    # GMRES will be run with diagonal preconditioning
    if weighting == 'diagonal':
        Dinv = get_diagonal(A, norm_eq=False, inv=True)
    elif weighting == 'block':
        Dinv = get_block_diag(A, blocksize=A.blocksize[0], inv_flag=True)
        Dinv = sparse.bsr_array((Dinv, np.arange(Dinv.shape[0], dtype=np.int32),
                                 np.arange(Dinv.shape[0] + 1, dtype=np.int32)),
                                 shape=A.shape)
    elif weighting == 'local':
        # Based on Gershgorin estimate
        D = np.abs(A)@np.ones((A.shape[0], 1), dtype=A.dtype)
        Dinv = np.zeros_like(D)
        Dinv[D != 0] = 1.0 / np.abs(D[D != 0])
    else:
        raise ValueError('weighting value is invalid')

    # Calculate initial residual
    #   Equivalent to R = -A@T;    R = R.multiply(pattern)
    #   with the added constraint that R has an explicit 0 wherever
    #   R is 0 and pattern is not
    uones = np.zeros(pattern.data.shape, dtype=T.dtype)
    R = sparse.bsr_array((uones, pattern.indices, pattern.indptr),
                          shape=pattern.shape)
    amg_core.incomplete_mat_mult_bsr(A.indptr, A.indices,
                                     np.ravel(A.data),
                                     T.indptr, T.indices,
                                     np.ravel(T.data),
                                     R.indptr, R.indices,
                                     np.ravel(R.data),
                                     int(T.shape[0]/T.blocksize[0]),
                                     int(T.shape[1]/T.blocksize[1]),
                                     A.blocksize[0], A.blocksize[1],
                                     T.blocksize[1])
    R.data *= -1.0

    # Apply diagonal preconditioner
    if weighting in ('local', 'diagonal'):
        R = scale_rows(R, Dinv)
    else:
        R = Dinv@R

    # Enforce R@B = 0
    satisfy_constraints(R, B, BtBinv)

    if R.nnz == 0:
        print('Error in sa_energy_min(..).  Initial R no nonzeros on a level. '
              'Returning tentative prolongator\n')
        return T

    # This is the RHS vector for the problem in the Krylov Space
    normr = np.sqrt((R.data.conjugate()*R.data).sum())
    g = np.zeros((maxiter+1,), dtype=xtype)
    g[0] = normr

    # First Krylov vector
    # V[0] = r/normr
    if normr > 0.0:
        V.append((1.0/normr)*R)

    # print "Energy Minimization of Prolongator \
    # --- Iteration 0 --- r = " + str(normr)
    i = -1
    # vect = np.ravel((A@T).data)
    # print "Iteration " + str(i+1) + "   \
    # Energy = %1.3e"%np.sqrt( (vect.conjugate()*vect).sum() )
    # print "Iteration " + str(i+1) + "   Normr  %1.3e"%normr
    while i < maxiter-1 and normr > tol:
        i = i+1

        # Calculate new search direction
        #   Equivalent to:  AV = A@V;    AV = AV.multiply(pattern)
        #   with the added constraint that explicit zeros are in AP wherever
        #   AP = 0 and pattern does not
        AV.data[:] = 0.0
        amg_core.incomplete_mat_mult_bsr(A.indptr, A.indices,
                                         np.ravel(A.data),
                                         V[i].indptr, V[i].indices,
                                         np.ravel(V[i].data),
                                         AV.indptr, AV.indices,
                                         np.ravel(AV.data),
                                         int(T.shape[0]/T.blocksize[0]),
                                         int(T.shape[1]/T.blocksize[1]),
                                         A.blocksize[0], A.blocksize[1],
                                         T.blocksize[1])

        if weighting in ('local', 'diagonal'):
            AV = scale_rows(AV, Dinv)
        else:
            AV = Dinv@AV

        # Enforce AV@B = 0
        satisfy_constraints(AV, B, BtBinv)
        V.append(AV.copy())

        # Modified Gram-Schmidt
        for j in range(i+1):
            # Frobenius inner-product
            H[j, i] = (V[j].conjugate().multiply(V[i+1])).sum()
            V[i+1] = V[i+1] - H[j, i]*V[j]

        # Frobenius Norm
        H[i+1, i] = np.sqrt((V[i+1].data.conjugate()*V[i+1].data).sum())

        # Check for breakdown
        if H[i+1, i] != 0.0:
            V[i+1] = (1.0 / H[i+1, i]) * V[i+1]

        # Apply previous Givens rotations to H
        if i > 0:
            apply_givens(Q, H[:, i], i)

        # Calculate and apply next complex-valued Givens Rotation
        if H[i+1, i] != 0:
            h1 = H[i, i]
            h2 = H[i+1, i]
            h1_mag = np.abs(h1)
            h2_mag = np.abs(h2)
            if h1_mag < h2_mag:
                mu = h1/h2
                tau = np.conjugate(mu)/np.abs(mu)
            else:
                mu = h2/h1
                tau = mu/np.abs(mu)

            denom = np.sqrt(h1_mag**2 + h2_mag**2)
            c = h1_mag/denom
            s = h2_mag*tau/denom
            Qblock = np.array([[c, np.conjugate(s)], [-s, c]], dtype=xtype)
            Q.append(Qblock)

            # Apply Givens Rotation to g,
            #   the RHS for the linear system in the Krylov Subspace.
            g[i:i+2] = np.dot(Qblock, g[i:i+2])

            # Apply effect of Givens Rotation to H
            H[i, i] = np.dot(Qblock[0, :], H[i:i+2, i])
            H[i+1, i] = 0.0

        normr = np.abs(g[i+1])
        # print "Iteration " + str(i+1) + "   Normr  %1.3e"%normr
    # End while loop

    # Find best update to x in Krylov Space, V.  Solve (i x i) system.
    if i != -1:
        y = la.solve(H[0:i+1, 0:i+1], g[0:i+1])
        for j in range(i+1):
            T = T + y[j]*V[j]

    # vect = np.ravel((A@T).data)
    # print "Final Iteration " + str(i) + "   \
    # Energy = %1.3e"%np.sqrt( (vect.conjugate()*vect).sum() )

    # Ensure identity at C-pts
    if Cpt_params[0]:
        T = Cpt_params[1]['I_F']@T + Cpt_params[1]['P_I']

    return T


def energy_prolongation_smoother(A, T, Atilde, B, Bf, Cpt_params,
                                 krylov='cg', maxiter=4, tol=1e-8,
                                 degree=1, weighting='local',
                                 prefilter=None, postfilter=None,
                                 force_fit_candidates=False):
    """Minimize the energy of the coarse basis functions (columns of T).

    Both root-node and non-root-node style prolongation smoothing is available,
    see Cpt_params description below.

    Parameters
    ----------
    A : csr_array, bsr_array
        Sparse NxN matrix.
    T : bsr_array
        Tentative prolongator, a NxM sparse matrix (M < N).
    Atilde : csr_array
        Strength of connection matrix.
    B : array
        Near-nullspace modes for coarse grid.  Has shape (M,k) where
        k is the number of coarse candidate vectors.
    Bf : array
        Near-nullspace modes for fine grid.  Has shape (N,k) where
        k is the number of coarse candidate vectors.
    Cpt_params : tuple
        Tuple of the form (bool, dict).  If the Cpt_params[0] = False, then the
        standard SA prolongation smoothing is carried out.  If True, then
        root-node style prolongation smoothing is carried out.  The dict must
        be a dictionary of parameters containing, (1) for P_I, P_I.T is the
        injection matrix for the Cpts, (2) I_F is an identity matrix for only the
        F-points (i.e. I, but with zero rows and columns for C-points) and I_C is
        the C-point analogue to I_F.  See Notes below for more information.
    krylov : str
        'cg' for SPD systems.  Solve A T = 0 in a constraint space with CG
        'cgnr' for nonsymmetric and/or indefinite systems.
        Solve A T = 0 in a constraint space with CGNR
        'gmres' for nonsymmetric and/or indefinite systems.
        Solve A T = 0 in a constraint space with GMRES.
    maxiter : int
        Number of energy minimization steps to apply to the prolongator.
    tol : scalar
        Minimization tolerance.
    degree : int
        Generate sparsity pattern for P based on (Atilde^degree T).
    weighting : str
        'block', 'diagonal' or 'local' construction of the diagonal preconditioning
        'local' Uses a local row-wise weight based on the Gershgorin estimate.
        Avoids any potential under-damping due to inaccurate spectral
        radius estimates.
        'block' Uses a block diagonal inverse of A if A is BSR.
        'diagonal' Uses the inverse of the diagonal of A.
    prefilter : dict
        Filter elements by row in sparsity pattern for P to reduce operator and
        setup complexity. If None or an empty dictionary, then no dropping in P
        is done.  If postfilter has key 'k', then the largest 'k' entries  are
        kept in each row.  If postfilter has key 'theta', all entries such that
        :math:`P[i,j] < kwargs['theta']*max(abs(P[i,:]))`
        are dropped.  If postfilter['k'] and postfiler['theta'] are present,
        then they are used with the union of their patterns.
    postfilter : dict
        Filters elements by row in smoothed P to reduce operator complexity.
        Only supported if using the rootnode_solver. If None or an empty
        dictionary, no dropping in P is done. If postfilter has key 'k',
        then the largest 'k' entries  are kept in each row.  If postfilter
        has key 'theta', all entries such that
        ``P[i,j] < kwargs['theta']*max(abs(P[i,:]))``
        are dropped.  If postfilter['k'] and postfiler['theta'] are present,
        then they are used with the union of their patterns.
    force_fit_candidates : boolean
        Use this option if the tentative T does not satisfy T B = Bf. 
        Default is False.

    Returns
    -------
    bsr_array
        Smoothed prolongator.

    Notes
    -----
    Only 'diagonal' weighting is supported for the CGNR method, because
    we are working with A^* A and not A.

    When Cpt_params[0] == True, root-node style prolongation smoothing is used
    to minimize the energy of columns of T.  Essentially, an identity block is
    maintained in T, corresponding to injection from the coarse-grid to the
    fine-grid root-nodes.  See [2]_ for more details, and see
    util.utils.get_Cpt_params for the helper function to generate Cpt_params.

    If Cpt_params[0] == False, the energy of columns of T are still
    minimized, but without maintaining the identity block.

    See [1]_ for more details on smoothed aggregation.

    References
    ----------
    .. [1] Jan Mandel, Marian Brezina, and Petr Vanek
       "Energy Optimization of Algebraic Multigrid Bases"
       Computing 62, 205-228, 1999
       http://dx.doi.org/10.1007/s006070050022

    .. [2] Olson, L. and Schroder, J. and Tuminaro, R.,
       "A general interpolation strategy for algebraic
       multigrid using energy minimization", SIAM Journal
       on Scientific Computing (SISC), vol. 33, pp.
       966--991, 2011.

    Examples
    --------
    >>> from pyamg.aggregation import energy_prolongation_smoother
    >>> from pyamg.gallery import poisson
    >>> from scipy.sparse import coo_array
    >>> import numpy as np
    >>> data = np.ones((6,))
    >>> row = np.arange(0, 6, dtype=np.int32)
    >>> col = np.kron([0, 1], np.ones(3)).astype(np.int32)
    >>> T = coo_array((data,(row, col)),shape=(6, 2)).tocsr()
    >>> print(T.toarray())
    [[1. 0.]
     [1. 0.]
     [1. 0.]
     [0. 1.]
     [0. 1.]
     [0. 1.]]
    >>> A = poisson((6,),format='csr')
    >>> B = np.ones((2,1),dtype=float)
    >>> P = energy_prolongation_smoother(A,T,A,B, None, (False,{}))
    >>> print(P.toarray())
    [[1.         0.        ]
     [1.         0.        ]
     [0.66666667 0.33333333]
     [0.33333333 0.66666667]
     [0.         1.        ]
     [0.         1.        ]]

    """
    # Test Inputs
    if maxiter < 0:
        raise ValueError('maxiter must be > 0')
    if tol > 1:
        raise ValueError('tol must be <= 1')

    if sparse.issparse(A) and A.format == 'csr':
        A = A.tobsr(blocksize=(1, 1), copy=False)
    elif sparse.issparse(A) and A.format == 'bsr':
        pass
    else:
        raise TypeError('A must be sparse BSR or CSR')

    if sparse.issparse(T) and T.format == 'csr':
        T = T.tobsr(blocksize=(1, 1), copy=False)
    elif sparse.issparse(T) and T.format == 'bsr':
        pass
    else:
        raise TypeError('T must be sparse BSR or CSR')

    if T.blocksize[0] != A.blocksize[0]:
        raise ValueError('T row-blocksize should be the same as A blocksize')

    if B.shape[0] != T.shape[1]:
        raise ValueError('B is the candidates for the coarse grid. '
                         '  num_rows(b) = num_cols(T)')

    if min(T.nnz, A.nnz) == 0:
        return T

    if not sparse.issparse(Atilde) or Atilde.format != 'csr':
        raise TypeError('Atilde must be csr_array')

    if prefilter is None:
        prefilter = {}

    if postfilter is None:
        postfilter = {}

    if ('theta' in prefilter) and (prefilter['theta'] == 0):
        prefilter.pop('theta', None)

    if ('theta' in postfilter) and (postfilter['theta'] == 0):
        postfilter.pop('theta', None)

    # Prepocess Atilde, the strength matrix
    if Atilde is None:
        Atilde = sparse.csr_array((np.ones(len(A.indices)),
                                    A.indices.copy(), A.indptr.copy()),
                                   shape=(A.shape[0]/A.blocksize[0],
                                          A.shape[1]/A.blocksize[1]))

    # If Atilde has no nonzeros, then return T
    if min(T.nnz, A.nnz) == 0:
        return T

    # Expand allowed sparsity pattern for P through multiplication by Atilde
    if degree > 0:

        # Construct pattern by multiplying with Atilde
        T.sort_indices()
        shape = (int(T.shape[0]/T.blocksize[0]),
                 int(T.shape[1]/T.blocksize[1]))
        pattern = sparse.csr_array((np.ones(T.indices.shape), T.indices, T.indptr),
                                    shape=shape)

        AtildeCopy = Atilde.copy()
        for _ in range(degree):
            pattern = AtildeCopy @ pattern

        # Optional filtering of sparsity pattern before smoothing
        if 'theta' in prefilter and 'k' in prefilter:
            pattern_theta = filter_matrix_rows(pattern, prefilter['theta'])
            pattern = truncate_rows(pattern, prefilter['k'])
            # Union two sparsity patterns
            pattern += pattern_theta
        elif 'k' in prefilter:
            pattern = truncate_rows(pattern, prefilter['k'])
        elif 'theta' in prefilter:
            pattern = filter_matrix_rows(pattern, prefilter['theta'])
        elif len(prefilter) > 0:
            raise ValueError('Unrecognized prefilter option')

        # unamal returns a BSR matrix with 1's in the nonzero locations
        pattern = unamal(pattern, T.blocksize[0], T.blocksize[1])
        pattern.sort_indices()

    else:
        # If degree is 0, just copy T for the sparsity pattern
        pattern = T.copy()
        if 'theta' in prefilter and 'k' in prefilter:
            pattern_theta = filter_matrix_rows(pattern, prefilter['theta'])
            pattern = truncate_rows(pattern, prefilter['k'])
            # Union two sparsity patterns
            pattern += pattern_theta
        elif 'k' in prefilter:
            pattern = truncate_rows(pattern, prefilter['k'])
        elif 'theta' in prefilter:
            pattern = filter_matrix_rows(pattern, prefilter['theta'])
        elif len(prefilter) > 0:
            raise ValueError('Unrecognized prefilter option')

        pattern.data[:] = 1.0
        pattern.sort_indices()

    # If using root nodes, enforce identity at C-points
    if Cpt_params[0]:
        pattern = Cpt_params[1]['I_F'] @ pattern
        pattern = Cpt_params[1]['P_I'] + pattern

    # Construct array of inv(Bi'Bi), where Bi is B restricted to row i's
    # sparsity pattern in pattern. This array is used multiple times
    # in satisfy_constraints(...).
    BtBinv = compute_BtBinv(B, pattern)

    # If using root nodes and B has more columns that A's blocksize, then
    # T must be updated so that T@B = Bfine.  Note, if this is a 'secondpass'
    # after dropping entries in P, then we must re-enforce the constraints
    if ((Cpt_params[0] and (B.shape[1] > A.blocksize[0]))
       or ('secondpass' in postfilter) or force_fit_candidates):
        T = filter_operator(T, pattern, B, Bf, BtBinv)
        # Ensure identity at C-pts
        if Cpt_params[0]:
            T = Cpt_params[1]['I_F']@T + Cpt_params[1]['P_I']

    # Iteratively minimize the energy of T subject to the constraints of
    # pattern and maintaining T's effect on B, i.e. T@B =
    # (T+Update)@B, i.e. Update@B = 0
    if krylov == 'cg':
        T = cg_prolongation_smoothing(A, T, B, BtBinv, pattern,
                                      maxiter, tol, weighting, Cpt_params)
    elif krylov == 'cgnr':
        T = cgnr_prolongation_smoothing(A, T, B, BtBinv, pattern,
                                        maxiter, tol, weighting, Cpt_params)
    elif krylov == 'gmres':
        T = gmres_prolongation_smoothing(A, T, B, BtBinv, pattern,
                                         maxiter, tol, weighting, Cpt_params)

    T.eliminate_zeros()

    # Filter entries in P, only in the rootnode case,
    # i.e., Cpt_params[0] == True
    if ((len(postfilter) == 0)
       or ('secondpass' in postfilter)
       or (Cpt_params[0] is False)):
        return T

    if 'theta' in postfilter and 'k' in postfilter:
        T_theta = filter_matrix_rows(T, postfilter['theta'])
        T_k = truncate_rows(T, postfilter['k'])

        # Union two sparsity patterns
        T_theta.data[:] = 1.0
        T_k.data[:] = 1.0
        T_filter = T_theta + T_k
        T_filter.data[:] = 1.0
        T_filter = T.multiply(T_filter)

    elif 'k' in postfilter:
        T_filter = truncate_rows(T, postfilter['k'])
    elif 'theta' in postfilter:
        T_filter = filter_matrix_rows(T, postfilter['theta'])
    else:
        raise ValueError('Unrecognized postfilter option')

    # Re-smooth T_filter and re-fit the modes B into the span.
    # Note, we set 'secondpass', because this is the second
    # filtering pass
    T = energy_prolongation_smoother(A, T_filter,
                                     Atilde, B, Bf, Cpt_params,
                                     krylov=krylov, maxiter=1,
                                     tol=1e-8, degree=0,
                                     weighting=weighting,
                                     prefilter={},
                                     postfilter={'secondpass': True})

    return T



def get_block_inverses(A, subdomain, subdomain_ptr):
    """
    Compute block inverses to A, where each block size is defined by subdomain
    and subdomain_ptr (CSR-like data structure)

    Parameters
    ----------
    A : {csr_matrix}


    Returns
    -------
    block_invs[i] = block inverse for subdomain i

    """

    class new_array(np.ndarray):
        ''' Hack of np array to allow for array.solve member function '''
        def __new__(cls, input_array):
            obj = np.asarray(input_array).view(cls)
            obj.solve = input_array.dot
            return obj

        def __array_finalize__(self, obj):
            if obj is None: return
            self.your_new_attr = getattr(obj, 'solve', None)

    ##
    # Extract each subdomain's block from the matrix
    inv_subblock_ptr = np.zeros(subdomain_ptr.shape,
                                dtype=A.indices.dtype)
    blocksize = (subdomain_ptr[1:] - subdomain_ptr[:-1])
    inv_subblock_ptr[1:] = np.cumsum(blocksize*blocksize)
    inv_subblock = np.zeros((inv_subblock_ptr[-1],), dtype=A.dtype)
    amg_core.extract_subblocks(A.indptr, A.indices, A.data, inv_subblock,
                               inv_subblock_ptr, subdomain, subdomain_ptr,
                               int(subdomain_ptr.shape[0]-1), A.shape[0])
    ##
    # Invert each block of A

    # Choose tolerance for which singular values are zero in *gelss below
    cond = set_tol(A.dtype)
    my_pinv, = lapack.get_lapack_funcs(['gelss'],
                                   (np.ones((1,), dtype=A.dtype)))
    for i in range(subdomain_ptr.shape[0]-1):
        m = blocksize[i]
        rhs = np.eye(m, m, dtype=A.dtype)
        j0 = inv_subblock_ptr[i]
        j1 = inv_subblock_ptr[i+1]
        if( j1 > j0):
            # sometimes for simple problems, the RHS can have a zero column 
            gelssoutput = my_pinv(inv_subblock[j0:j1].reshape(m, m),
                                  rhs, cond=cond, overwrite_a=True,
                                  overwrite_b=True)
            inv_subblock[j0:j1] = np.ravel(gelssoutput[1])

    ##
    # Convert inverted blocks into a list of little inverses ready for use
    block_invs = np.array_split(inv_subblock, inv_subblock_ptr[1:-1])
    block_invs = [ new_array(block.reshape(bsize,bsize)) for (block, bsize) in zip(block_invs, blocksize) ]

    return block_invs


def get_block_diag_inv(A, subdomain, subdomain_ptr):
    """
    Compute approximate block inverses to A, using just the diagonal of A,
    where each block size is defined by subdomain and subdomain_ptr (CSR-like
    data structure)

    Parameters
    ----------
    A : {csr_matrix}


    Returns
    -------
    block_invs[i] = approximate block inverse for subdomain i

    """

    class new_array(np.ndarray):
        ''' Hack of np array to allow for array.solve member function '''
        def __new__(cls, input_array):
            obj = np.asarray(input_array).view(cls)
            obj.solve = input_array.__mul__
            return obj

        def __array_finalize__(self, obj):
            if obj is None: return
            self.your_new_attr = getattr(obj, 'solve', None)

    ##
    # Extract diagonal of A
    D = (A.tocsr()).diagonal()

    ##
    # Grab each block's part of D
    block_invs = []
    for i in range(subdomain_ptr.shape[0]-1):
        start = subdomain_ptr[i]
        end = subdomain_ptr[i+1]
        local_D_inv = D[ subdomain[start:end] ]
        local_D_inv[ local_D_inv != 0] = 1.0 / local_D_inv[ local_D_inv != 0]
        block_invs.append( new_array(local_D_inv) )

    return block_invs



def AIRplus(A, B, Bf, Cpt_params, InitialPattern='A_FC',  maxiter=1, degree=1, block_solver='exact', strength='transpose', theta=0.05, norm='abs', AggOp=None, **kwargs):
    """Solve the AIR equations 
             A_FF^{-1} W = -A_FC 
       subject to mode interpolation constraints.

    This differs from the rootnode solver, in part, because it allows for
    separate, local inverses when computing updates for column i of W.
    Essentially, each column i of W corresponds to (approx) inverting a window
    of A_FF, and here each (approx) inverse of an A_FF window is computed
    independently.  For rootnode, a single global Krylov polynomial is used to
    compute all inverses of A_FF windows. 

    Parameters
    ----------
    A : csr_matrix, bsr_matrix
        Sparse NxN matrix
    B : array
        Near-nullspace modes for coarse grid.  Has shape (M,k) where
        k is the number of coarse candidate vectors.
    Bf : array
        Near-nullspace modes for fine grid.  Has shape (N,k) where
        k is the number of coarse candidate vectors.
    Cpt_params : tuple
        Tuple of the form (bool, dict).  If the Cpt_params[0] = False, then the
        standard SA prolongation smoothing is carried out.  If True, then
        root-node style prolongation smoothing is carried out.  The dict must
        be a dictionary of parameters containing, (1) for P_I, P_I.T is the
        injection matrix for the Cpts, (2) I_F is an identity matrix for only the
        F-points (i.e. I, but with zero rows and columns for C-points) and I_C is
        the C-point analogue to I_F.  See Notes below for more information.
    InitialPattern : string, sparse matrix
        'A_FC', then [ -A_FC, I ] is used as the initial pattern (based on strong connections)
        'Aggregation', then use AggOp as the initial pattern
        Sparse Matrix, diagnostic mode that uses a preset matrix as the patter (two-level only)
    maxiter : integer
        Number of iterations when solving for W 
    degree : int
        Beginning sparsity pattern for P based on ( Atilde^degree  [-A_FC;  I] )
        The degree should be > 0 when you need more nonzeros to satisfy the constraints
        in the initial P.  Atilde is the strength of connection matrix.
    block_solver : string
        Type of solver to use when solving the local linear systems for each column of P via AIR
        'exact' - use pseudoinverse of the block
        'diag'  - use just the diagonal of the block
        'splu' [to implement]
        'tria' [to implement, choose upper or lower triag and use that]
    strength : string
        'transpose' or 'no-transpose'
        Decides whether to compute strength based on A or A.H
        Note, for restriction, A.H is already passed in.
        Note, for classic AIR, 'transpose' is the standard option.
    theta : float
        Strength of connection tolerance for computing interpolation neighborhood
        See classical_strength_of_connection
    norm : string
        Typically 'abs' or 'min', used by strength of connection in block case
        See classical_strength_of_connection
    AggOp : None, CSR matrix
        When InitialPattern='Aggregation', AggOp is used as the initial interpolation pattern


    Returns
    -------
    T : bsr_matrix
        Smoothed prolongator
    """

    # TODO: this code doesn't support BSR A.  Not sure how to handle this case.
    #       It may be that if the coarsening is done block-wise, followed by the
    #       C-pts and F-pts being unamalgamated to regular DOFs, then this code 
    #       will do something reasonable

    # Test Inputs
    if maxiter < 1:
        raise ValueError('maxiter must be > 1')
    if degree < 0:
        raise ValueError('degree must be > 0')

    # Convert A to CSR
    if sparse.isspmatrix_csr(A):
        pass
    elif sparse.isspmatrix_bsr(A):
        A = A.tocsr(copy=False)
    else:
        raise TypeError('A must be csr_matrix or bsr_matrix')
    if A.nnz == 0:
        raise TypeError('A must have nonzeros')

    ##
    # Generate SOC
    if strength == 'transpose':
        # This will match the Atilde used by classic AIR
        #   AIR uses Atilde[C,F], where Atilde is built based on A.  Since we
        #   pass in A.T to this routine, these two lines will match the behavior
        #   of AIR, but for an efficient implementation, we probably want to
        #   avoid these transposes.
        Atilde = classical_strength_of_connection(A=A.T.tocsr(), theta=theta, block=False, norm=norm)
        Atilde = Atilde.T.tocsr()
    else:
        Atilde = classical_strength_of_connection(A=A, theta=theta, block=False, norm=norm)

    ##
    # Extract needed C/F splitting information
    I_F = Cpt_params[1]['I_F'].tocsr()
    I_C = Cpt_params[1]['I_C'].tocsr()
    P_I = Cpt_params[1]['P_I'].tocsr()
    Cpts = Cpt_params[1]['Cpts']
    AFF = I_F @ A @ I_F
    AFC = ((I_F@A)[:,Cpts]).tocsr()            # Used to compute residual for AIR-type operator
    AtildeFC = ((I_F@Atilde)[:,Cpts]).tocsr()  # Used for sparsity pattern (to match classic AIR)

    ##
    # If no Cpts, then return zero matrix (P_I)
    if Cpts.shape[0] == 0:
        return P_I

    ##
    # Initialize T = [- A_FC; I]
    T = -AFC + P_I

    ##
    # Compute initial sparsity pattern, using preset or [ -A_FC; I]
    if sparse.isspmatrix_csr(InitialPattern) or sparse.isspmatrix_bsr(InitialPattern):
        if InitialPattern.shape[0] < InitialPattern.shape[1]:
            pattern = InitialPattern.T.tocsr().copy()
        else:
            pattern = InitialPattern.tocsr().copy()
    elif InitialPattern == "Aggregation":
        if sparse.isspmatrix_csr(AggOp):
            pattern = AggOp.copy()
        else:
            raise ValueError(f"Aggregation for InitialPattern is only compatible with aggregation-based C/F splittings, like 'standard' or 'naive'")
    elif InitialPattern == "A_FC":
        pattern = AtildeFC + P_I
    else:
        raise ValueError(f"InitialPattern {InitialPattern} is incorrect.  It should be a matrix of compatible dimension, 'Aggregation', or 'A_FC'")

    ##
    # Expanding the pattern through multiplication by Atilde
    pattern.data[:] = 1.0
    for _ in range(degree):
        pattern = Atilde @ pattern
        #pattern = (I_F@Atilde@I_F) @ pattern

    #
    pattern.data[:] = 1.0
    #
    # Enforce identity at C-points
    pattern = I_F@pattern
    pattern = P_I + pattern
    num_ff_nnz = (I_F@pattern).nnz
    pattern.sort_indices()

    ##
    # Compute data for constraint enforcement (BtBinv) and block_invs for
    # solving (inexactly) Aff W = -Afc. Similar to classic en-min, we treat the
    # above pattern as fixed, and compute these quantities _only_ once 
    #
    # Construct array of inv(Bi'Bi), where Bi is B restricted to row i's
    # sparsity pattern in pattern. Needed by satisfy_constraints(...)
    BtBinv = compute_BtBinv(B, pattern)
    #
    # Compute (approx) block inverses for corresponding windows of A_FF
    pattern_FF= (I_F@pattern).tocsc()
    pattern_FF.sort_indices()
    if block_solver == 'exact':
        block_invs = get_block_inverses(A, pattern_FF.indices, pattern_FF.indptr)
    elif block_solver == 'diag':
        block_invs = get_block_diag_inv(A, pattern_FF.indices, pattern_FF.indptr)
    else:
        raise ValueError(f"Invalid block_solver option {block_solver}")

    ##
    # Update T to satisfy the constraints (T@B = Bf), have identity at C-pts, and the 
    # same sparsity pattern as "pattern"
    T = filter_operator(T, pattern, B, Bf, BtBinv)     # Comment out this line to match output from classic AIR
    T = I_F@T + P_I
    T = T.multiply(pattern) + 1e-12*pattern_FF
    T.data[ T.data == 1e-12] = 0.0
    T.sort_indices()
    if(T.nnz != pattern.nnz):
        raise ValueError('T and pattern should have same nnz, perhaps some values canceled during computation of T')
    
    ##
    # Iteratively solve the AIR equations
    #    A_FF W_update = - A_FC - A_FF W_T
    # where W_T is the weight block from T and each iteration, W_update is
    # projected so that W_update@B = 0.  Note, the original W satisfies the
    # constraints.
    for i in range(maxiter):
        # Compute right-hand-side, making sure to have same nonzero pattern as "pattern"
        RHS = -AFC - AFF@T
        if RHS.data.ravel().shape[0] > 0:
            smallest = 1e-12*np.abs(RHS.data.ravel()).min()
        elif A.data.ravel().shape[0] > 0:
            smallest = 1e-12*np.abs(A.data.ravel()).min()
        else:
            smallest = 1e-12
        RHS = RHS.multiply(pattern) + smallest*pattern_FF
        RHS.data[ RHS.data == smallest] = 0.0

        if(RHS.nnz != num_ff_nnz):
            raise ValueError('RHS and W should have same nnz, perhaps some values canceled during computation of T')
        
        # Compute interpolation update T_hat
        RHS = RHS.tocsc()
        RHS.sort_indices()
        T_hat = RHS.copy()
        ## RHS, T_hat, and pattern_FF should have the exact same sparsity pattern and indices at this point
        for k, (start,end) in enumerate( zip(RHS.indptr[:-1], RHS.indptr[1:]) ):
            T_hat.data[start:end] = block_invs[k].solve(RHS.data[start:end])
            #T_hat.data[start:end] = 0.25*RHS.data[start:end]
        #
        T_hat = T_hat.tobsr(blocksize=(1,1))

        # Project update so that T_hat@B = 0
        satisfy_constraints(T_hat, B, BtBinv)      # Comment out this line to match output from classic AIR
        # Add update to T
        T = T + T_hat.tocsr()
    
    T = T.tobsr(blocksize=(1,1))
    return T


