"""
Strength of Connection functions

Requirements for the strength matrix C are:
    1) Nonzero diagonal whenever A has a nonzero diagonal
    2) Non-negative entries (float or bool) in [0,1]
    3) Large entries denoting stronger connections
    4) C denotes nodal connections, i.e., if A is an nxn BSR matrix with
       row block size of m, then C is (n/m) x (n/m)

"""
from __future__ import print_function


from warnings import warn

import numpy as np
from pyamg.util.utils import scale_rows_by_largest_entry, amalgamate
from scipy import sparse
from pyamg import amg_core
from pyamg.relaxation.relaxation import jacobi

__all__ = ['classical_strength_of_connection',
           'symmetric_strength_of_connection',
           'evolution_strength_of_connection',
           'distance_strength_of_connection',
           'algebraic_distance',
           'affinity_distance',
           # deprecated:
           'ode_strength_of_connection']


def distance_strength_of_connection(A, V, theta=2.0, relative_drop=True):
    """
    Distance based strength-of-connection

    Parameters
    ----------
    A : csr_matrix or bsr_matrix
        Square, sparse matrix in CSR or BSR format
    V : array
        Coordinates of the vertices of the graph of A
    relative_drop : bool
        If false, then a connection must be within a distance of theta
        from a point to be strongly connected.
        If true, then the closest connection is always strong, and other points
        must be within theta times the smallest distance to be strong

    Returns
    -------
    C : csr_matrix
        C(i,j) = distance(point_i, point_j)
        Strength of connection matrix where strength values are
        distances, i.e. the smaller the value, the stronger the connection.
        Sparsity pattern of C is copied from A.

    Notes
    -----
    - theta is a drop tolerance that is applied row-wise
    - If a BSR matrix given, then the return matrix is still CSR.  The strength
      is given between super nodes based on the BSR block size.

    Examples
    --------
    >>> from pyamg.gallery import load_example
    >>> from pyamg.strength import distance_strength_of_connection
    >>> data = load_example('airfoil')
    >>> A = data['A'].tocsr()
    >>> S = distance_strength_of_connection(data['A'], data['vertices'])

    """
    # Amalgamate for the supernode case
    if sparse.isspmatrix_bsr(A):
        sn = int(A.shape[0] / A.blocksize[0])
        u = np.ones((A.data.shape[0],))
        A = sparse.csr_matrix((u, A.indices, A.indptr), shape=(sn, sn))

    if not sparse.isspmatrix_csr(A):
        warn("Implicit conversion of A to csr", sparse.SparseEfficiencyWarning)
        A = sparse.csr_matrix(A)

    dim = V.shape[1]

    # Create two arrays for differencing the different coordinates such
    # that C(i,j) = distance(point_i, point_j)
    cols = A.indices
    rows = np.repeat(np.arange(A.shape[0]), A.indptr[1:] - A.indptr[0:-1])

    # Insert difference for each coordinate into C
    C = (V[rows, 0] - V[cols, 0])**2
    for d in range(1, dim):
        C += (V[rows, d] - V[cols, d])**2
    C = np.sqrt(C)
    C[C < 1e-6] = 1e-6

    C = sparse.csr_matrix((C, A.indices.copy(), A.indptr.copy()),
                          shape=A.shape)

    # Apply drop tolerance
    if relative_drop is True:
        if theta != np.inf:
            amg_core.apply_distance_filter(C.shape[0], theta, C.indptr,
                                           C.indices, C.data)
    else:
        amg_core.apply_absolute_distance_filter(C.shape[0], theta, C.indptr,
                                                C.indices, C.data)
    C.eliminate_zeros()

    C = C + sparse.eye(C.shape[0], C.shape[1], format='csr')

    # Standardized strength values require small values be weak and large
    # values be strong.  So, we invert the distances.
    C.data = 1.0 / C.data

    # Scale C by the largest magnitude entry in each row
    C = scale_rows_by_largest_entry(C)

    return C


def classical_strength_of_connection(A, theta=0.0, norm='abs'):
    """
    Return a strength of connection matrix using the classical AMG measure
    An off-diagonal entry A[i,j] is a strong connection iff::

             A[i,j] >= theta * max(|A[i,k]|), where k != i     (norm='abs')
            -A[i,j] >= theta * max(-A[i,k]),  where k != i     (norm='min')

    Parameters
    ----------
    A : csr_matrix or bsr_matrix
        Square, sparse matrix in CSR or BSR format
    theta : float
        Threshold parameter in [0,1].
    norm: 'string'
        'abs' : to use the absolute value,
        'min' : to use the negative value (see above)

    Returns
    -------
    S : csr_matrix
        Matrix graph defining strong connections.  S[i,j]=1 if vertex i
        is strongly influenced by vertex j.

    See Also
    --------
    symmetric_strength_of_connection : symmetric measure used in SA
    evolution_strength_of_connection : relaxation based strength measure

    Notes
    -----
    - A symmetric A does not necessarily yield a symmetric strength matrix S
    - Calls C++ function classical_strength_of_connection
    - The version as implemented is designed form M-matrices.  Trottenberg et
      al. use max A[i,k] over all negative entries, which is the same.  A
      positive edge weight never indicates a strong connection.
    - See [2000BrHeMc]_ and [2001bTrOoSc]_

    References
    ----------

    .. [2000BrHeMc] Briggs, W. L., Henson, V. E., McCormick, S. F., "A multigrid
        tutorial", Second edition. Society for Industrial and Applied
        Mathematics (SIAM), Philadelphia, PA, 2000. xii+193 pp.

    .. [2001bTrOoSc] Trottenberg, U., Oosterlee, C. W., Schuller, A., "Multigrid",
        Academic Press, Inc., San Diego, CA, 2001. xvi+631 pp.

    Examples
    --------
    >>> import numpy as np
    >>> from pyamg.gallery import stencil_grid
    >>> from pyamg.strength import classical_strength_of_connection
    >>> n=3
    >>> stencil = np.array([[-1.0,-1.0,-1.0],
    ...                        [-1.0, 8.0,-1.0],
    ...                        [-1.0,-1.0,-1.0]])
    >>> A = stencil_grid(stencil, (n,n), format='csr')
    >>> S = classical_strength_of_connection(A, 0.0)

    """

    if sparse.isspmatrix_bsr(A):
        blocksize = A.blocksize[0]
    else:
        blocksize = 1

    if not sparse.isspmatrix_csr(A):
        warn("Implicit conversion of A to csr", sparse.SparseEfficiencyWarning)
        A = sparse.csr_matrix(A)

    if (theta < 0 or theta > 1):
        raise ValueError('expected theta in [0,1]')

    Sp = np.empty_like(A.indptr)
    Sj = np.empty_like(A.indices)
    Sx = np.empty_like(A.data)

    if norm == 'abs':
        amg_core.classical_strength_of_connection_abs(A.shape[0], theta,
                                                      A.indptr, A.indices, A.data,
                                                      Sp, Sj, Sx)
    elif norm == 'min':
        amg_core.classical_strength_of_connection_min(A.shape[0], theta,
                                                      A.indptr, A.indices, A.data,
                                                      Sp, Sj, Sx)
    else:
        raise ValueError('Unknown norm')

    S = sparse.csr_matrix((Sx, Sj, Sp), shape=A.shape)

    if blocksize > 1:
        S = amalgamate(S, blocksize)

    # Strength represents "distance", so take the magnitude
    S.data = np.abs(S.data)

    # Scale S by the largest magnitude entry in each row
    S = scale_rows_by_largest_entry(S)

    return S


def symmetric_strength_of_connection(A, theta=0):
    """
    Compute strength of connection matrix using the standard symmetric measure

    An off-diagonal connection A[i,j] is strong iff::

        abs(A[i,j]) >= theta * sqrt( abs(A[i,i]) * abs(A[j,j]) )

    Parameters
    ----------
    A : csr_matrix
        Matrix graph defined in sparse format.  Entry A[i,j] describes the
        strength of edge [i,j]
    theta : float
        Threshold parameter (positive).

    Returns
    -------
    S : csr_matrix
        Matrix graph defining strong connections.  S[i,j]=1 if vertex i
        is strongly influenced by vertex j.

    See Also
    --------
    symmetric_strength_of_connection : symmetric measure used in SA
    evolution_strength_of_connection : relaxation based strength measure

    Notes
    -----
        - For vector problems, standard strength measures may produce
          undesirable aggregates.  A "block approach" from Vanek et al. is used
          to replace vertex comparisons with block-type comparisons.  A
          connection between nodes i and j in the block case is strong if::

          ||AB[i,j]|| >= theta * sqrt( ||AB[i,i]||*||AB[j,j]|| ) where AB[k,l]

          is the matrix block (degrees of freedom) associated with nodes k and
          l and ||.|| is a matrix norm, such a Frobenius.

        - See [1996bVaMaBr]_ for more details.

    References
    ----------
    .. [1996bVaMaBr] Vanek, P. and Mandel, J. and Brezina, M.,
       "Algebraic Multigrid by Smoothed Aggregation for
       Second and Fourth Order Elliptic Problems",
       Computing, vol. 56, no. 3, pp. 179--196, 1996.
       http://citeseer.ist.psu.edu/vanek96algebraic.html

    Examples
    --------
    >>> import numpy as np
    >>> from pyamg.gallery import stencil_grid
    >>> from pyamg.strength import symmetric_strength_of_connection
    >>> n=3
    >>> stencil = np.array([[-1.0,-1.0,-1.0],
    ...                        [-1.0, 8.0,-1.0],
    ...                        [-1.0,-1.0,-1.0]])
    >>> A = stencil_grid(stencil, (n,n), format='csr')
    >>> S = symmetric_strength_of_connection(A, 0.0)
    """

    if theta < 0:
        raise ValueError('expected a positive theta')

    if sparse.isspmatrix_csr(A):
        # if theta == 0:
        #     return A

        Sp = np.empty_like(A.indptr)
        Sj = np.empty_like(A.indices)
        Sx = np.empty_like(A.data)

        fn = amg_core.symmetric_strength_of_connection
        fn(A.shape[0], theta, A.indptr, A.indices, A.data, Sp, Sj, Sx)

        S = sparse.csr_matrix((Sx, Sj, Sp), shape=A.shape)

    elif sparse.isspmatrix_bsr(A):
        M, N = A.shape
        R, C = A.blocksize

        if R != C:
            raise ValueError('matrix must have square blocks')

        if theta == 0:
            data = np.ones(len(A.indices), dtype=A.dtype)
            S = sparse.csr_matrix((data, A.indices.copy(), A.indptr.copy()),
                                  shape=(int(M / R), int(N / C)))
        else:
            # the strength of connection matrix is based on the
            # Frobenius norms of the blocks
            data = (np.conjugate(A.data) * A.data).reshape(-1, R * C)
            data = data.sum(axis=1)
            A = sparse.csr_matrix((data, A.indices, A.indptr),
                                  shape=(int(M / R), int(N / C)))
            return symmetric_strength_of_connection(A, theta)
    else:
        raise TypeError('expected csr_matrix or bsr_matrix')

    # Strength represents "distance", so take the magnitude
    S.data = np.abs(S.data)

    # Scale S by the largest magnitude entry in each row
    S = scale_rows_by_largest_entry(S)

    return S


def energy_based_strength_of_connection(A, theta=0.0, k=2):
    """
    Compute a strength of connection matrix using an energy-based measure.

    Parameters
    ----------
    A : {sparse-matrix}
        matrix from which to generate strength of connection information
    theta : {float}
        Threshold parameter in [0,1]
    k : {int}
        Number of relaxation steps used to generate strength information

    Returns
    -------
    S : {csr_matrix}
        Matrix graph defining strong connections.  The sparsity pattern
        of S matches that of A.  For BSR matrices, S is a reduced strength
        of connection matrix that describes connections between supernodes.

    Notes
    -----
    This method relaxes with weighted-Jacobi in order to approximate the
    matrix inverse.  A normalized change of energy is then used to define
    point-wise strength of connection values.  Specifically, let v be the
    approximation to the i-th column of the inverse, then

    (S_ij)^2 = <v_j, v_j>_A / <v, v>_A,

    where v_j = v, such that entry j in v has been zeroed out.  As is common,
    larger values imply a stronger connection.

    Current implementation is a very slow pure-python implementation for
    experimental purposes, only.

    See [2006BrBrMaMaMc]_ for more details.

    References
    ----------
    .. [2006BrBrMaMaMc] Brannick, Brezina, MacLachlan, Manteuffel, McCormick.
       "An Energy-Based AMG Coarsening Strategy",
       Numerical Linear Algebra with Applications,
       vol. 13, pp. 133-148, 2006.

    Examples
    --------
    >>> import numpy as np
    >>> from pyamg.gallery import stencil_grid
    >>> from pyamg.strength import energy_based_strength_of_connection
    >>> n=3
    >>> stencil =  np.array([[-1.0,-1.0,-1.0],
    ...                        [-1.0, 8.0,-1.0],
    ...                        [-1.0,-1.0,-1.0]])
    >>> A = stencil_grid(stencil, (n,n), format='csr')
    >>> S = energy_based_strength_of_connection(A, 0.0)
    """

    if (theta < 0):
        raise ValueError('expected a positive theta')
    if not sparse.isspmatrix(A):
        raise ValueError('expected sparse matrix')
    if (k < 0):
        raise ValueError('expected positive number of steps')
    if not isinstance(k, int):
        raise ValueError('expected integer')

    if sparse.isspmatrix_bsr(A):
        bsr_flag = True
        numPDEs = A.blocksize[0]
        if A.blocksize[0] != A.blocksize[1]:
            raise ValueError('expected square blocks in BSR matrix A')
    else:
        bsr_flag = False

    # Convert A to csc and Atilde to csr
    if sparse.isspmatrix_csr(A):
        Atilde = A.copy()
        A = A.tocsc()
    else:
        A = A.tocsc()
        Atilde = A.copy()
        Atilde = Atilde.tocsr()

    # Calculate the weighted-Jacobi parameter
    from pyamg.util.linalg import approximate_spectral_radius
    D = A.diagonal()
    Dinv = 1.0 / D
    Dinv[D == 0] = 0.0
    Dinv = sparse.csc_matrix((Dinv, (np.arange(A.shape[0]),
                             np.arange(A.shape[1]))), shape=A.shape)
    DinvA = Dinv * A
    omega = 1.0 / approximate_spectral_radius(DinvA)
    del DinvA

    # Approximate A-inverse with k steps of w-Jacobi and a zero initial guess
    S = sparse.csc_matrix(A.shape, dtype=A.dtype)  # empty matrix
    Id = sparse.eye(A.shape[0], A.shape[1], format='csc')
    for i in range(k + 1):
        S = S + omega * (Dinv * (Id - A * S))

    # Calculate the strength entries in S column-wise, but only strength
    # values at the sparsity pattern of A
    for i in range(Atilde.shape[0]):
        v = np.mat(S[:, i].todense())
        Av = np.mat(A * v)
        denom = np.sqrt(np.conjugate(v).T * Av)
        # replace entries in row i with strength values
        for j in range(Atilde.indptr[i], Atilde.indptr[i + 1]):
            col = Atilde.indices[j]
            vj = v[col].copy()
            v[col] = 0.0
            #   =  (||v_j||_A - ||v||_A) / ||v||_A
            val = np.sqrt(np.conjugate(v).T * A * v) / denom - 1.0

            # Negative values generally imply a weak connection
            if val > -0.01:
                Atilde.data[j] = abs(val)
            else:
                Atilde.data[j] = 0.0

            v[col] = vj

    # Apply drop tolerance
    Atilde = classical_strength_of_connection(Atilde, theta=theta)
    Atilde.eliminate_zeros()

    # Put ones on the diagonal
    Atilde = Atilde + Id.tocsr()
    Atilde.sort_indices()

    # Amalgamate Atilde for the BSR case, using ones for all strong connections
    if bsr_flag:
        Atilde = Atilde.tobsr(blocksize=(numPDEs, numPDEs))
        nblocks = Atilde.indices.shape[0]
        uone = np.ones((nblocks,))
        Atilde = sparse.csr_matrix((uone, Atilde.indices, Atilde.indptr),
                                   shape=(
                                       int(Atilde.shape[0] / numPDEs),
                                       int(Atilde.shape[1] / numPDEs)))

    # Scale C by the largest magnitude entry in each row
    Atilde = scale_rows_by_largest_entry(Atilde)

    return Atilde


@np.deprecate
def ode_strength_of_connection(A, B=None, epsilon=4.0, k=2, proj_type="l2",
                               block_flag=False, symmetrize_measure=True):
    """Use evolution_strength_of_connection instead"""
    return evolution_strength_of_connection(A, B, epsilon, k, proj_type,
                                            block_flag, symmetrize_measure)


def evolution_strength_of_connection(A, B=None, epsilon=4.0, k=2,
                                     proj_type="l2", block_flag=False,
                                     symmetrize_measure=True):
    """
    Construct strength of connection matrix using an Evolution-based measure

    Parameters
    ----------
    A : {csr_matrix, bsr_matrix}
        Sparse NxN matrix
    B : {string, array}
        If B=None, then the near nullspace vector used is all ones.  If B is
        an (NxK) array, then B is taken to be the near nullspace vectors.
    epsilon : scalar
        Drop tolerance
    k : integer
        ODE num time steps, step size is assumed to be 1/rho(DinvA)
    proj_type : {'l2','D_A'}
        Define norm for constrained min prob, i.e. define projection
    block_flag : {boolean}
        If True, use a block D inverse as preconditioner for A during
        weighted-Jacobi

    Returns
    -------
    Atilde : {csr_matrix}
        Sparse matrix of strength values

    See [2008OlScTu]_ for more details.

    References
    ----------
    .. [2008OlScTu] Olson, L. N., Schroder, J., Tuminaro, R. S.,
       "A New Perspective on Strength Measures in Algebraic Multigrid",
       submitted, June, 2008.

    Examples
    --------
    >>> import numpy as np
    >>> from pyamg.gallery import stencil_grid
    >>> from pyamg.strength import evolution_strength_of_connection
    >>> n=3
    >>> stencil =  np.array([[-1.0,-1.0,-1.0],
    ...                        [-1.0, 8.0,-1.0],
    ...                        [-1.0,-1.0,-1.0]])
    >>> A = stencil_grid(stencil, (n,n), format='csr')
    >>> S = evolution_strength_of_connection(A,  np.ones((A.shape[0],1)))
    """
    # local imports for evolution_strength_of_connection
    from pyamg.util.utils import scale_rows, get_block_diag, scale_columns
    from pyamg.util.linalg import approximate_spectral_radius

    # ====================================================================
    # Check inputs
    if epsilon < 1.0:
        raise ValueError("expected epsilon > 1.0")
    if k <= 0:
        raise ValueError("number of time steps must be > 0")
    if proj_type not in ['l2', 'D_A']:
        raise ValueError("proj_type must be 'l2' or 'D_A'")
    if (not sparse.isspmatrix_csr(A)) and (not sparse.isspmatrix_bsr(A)):
        raise TypeError("expected csr_matrix or bsr_matrix")

    # ====================================================================
    # Format A and B correctly.
    # B must be in mat format, this isn't a deep copy
    if B is None:
        Bmat = np.mat(np.ones((A.shape[0], 1), dtype=A.dtype))
    else:
        Bmat = np.mat(B)

    # Pre-process A.  We need A in CSR, to be devoid of explicit 0's and have
    # sorted indices
    if (not sparse.isspmatrix_csr(A)):
        csrflag = False
        numPDEs = A.blocksize[0]
        D = A.diagonal()
        # Calculate Dinv*A
        if block_flag:
            Dinv = get_block_diag(A, blocksize=numPDEs, inv_flag=True)
            Dinv = sparse.bsr_matrix((Dinv, np.arange(Dinv.shape[0]),
                                     np.arange(Dinv.shape[0] + 1)),
                                     shape=A.shape)
            Dinv_A = (Dinv * A).tocsr()
        else:
            Dinv = np.zeros_like(D)
            mask = (D != 0.0)
            Dinv[mask] = 1.0 / D[mask]
            Dinv[D == 0] = 1.0
            Dinv_A = scale_rows(A, Dinv, copy=True)
        A = A.tocsr()
    else:
        csrflag = True
        numPDEs = 1
        D = A.diagonal()
        Dinv = np.zeros_like(D)
        mask = (D != 0.0)
        Dinv[mask] = 1.0 / D[mask]
        Dinv[D == 0] = 1.0
        Dinv_A = scale_rows(A, Dinv, copy=True)

    A.eliminate_zeros()
    A.sort_indices()

    # Handle preliminaries for the algorithm
    dimen = A.shape[1]
    NullDim = Bmat.shape[1]

    # Get spectral radius of Dinv*A, this will be used to scale the time step
    # size for the ODE
    rho_DinvA = approximate_spectral_radius(Dinv_A)

    # Calculate D_A for later use in the minimization problem
    if proj_type == "D_A":
        D_A = sparse.spdiags([D], [0], dimen, dimen, format='csr')
    else:
        D_A = sparse.eye(dimen, dimen, format="csr", dtype=A.dtype)

    # Calculate (I - delta_t Dinv A)^k
    #      In order to later access columns, we calculate the transpose in
    #      CSR format so that columns will be accessed efficiently
    # Calculate the number of time steps that can be done by squaring, and
    # the number of time steps that must be done incrementally
    nsquare = int(np.log2(k))
    ninc = k - 2**nsquare

    # Calculate one time step
    Id = sparse.eye(dimen, dimen, format="csr", dtype=A.dtype)
    Atilde = (Id - (1.0 / rho_DinvA) * Dinv_A)
    Atilde = Atilde.T.tocsr()

    # Construct a sparsity mask for Atilde that will restrict Atilde^T to the
    # nonzero pattern of A, with the added constraint that row i of Atilde^T
    # retains only the nonzeros that are also in the same PDE as i.
    mask = A.copy()

    # Restrict to same PDE
    if numPDEs > 1:
        row_length = np.diff(mask.indptr)
        my_pde = np.mod(np.arange(dimen), numPDEs)
        my_pde = np.repeat(my_pde, row_length)
        mask.data[np.mod(mask.indices, numPDEs) != my_pde] = 0.0
        del row_length, my_pde
        mask.eliminate_zeros()

    # If the total number of time steps is a power of two, then there is
    # a very efficient computational short-cut.  Otherwise, we support
    # other numbers of time steps, through an inefficient algorithm.
    if ninc > 0:
        warn("The most efficient time stepping for the Evolution Strength\
             Method is done in powers of two.\nYou have chosen " + str(k) +
             " time steps.")

        # Calculate (Atilde^nsquare)^T = (Atilde^T)^nsquare
        for i in range(nsquare):
            Atilde = Atilde * Atilde

        JacobiStep = (Id - (1.0 / rho_DinvA) * Dinv_A).T.tocsr()
        for i in range(ninc):
            Atilde = Atilde * JacobiStep
        del JacobiStep

        # Apply mask to Atilde, zeros in mask have already been eliminated at
        # start of routine.
        mask.data[:] = 1.0
        Atilde = Atilde.multiply(mask)
        Atilde.eliminate_zeros()
        Atilde.sort_indices()

    elif nsquare == 0:
        if numPDEs > 1:
            # Apply mask to Atilde, zeros in mask have already been eliminated
            # at start of routine.
            mask.data[:] = 1.0
            Atilde = Atilde.multiply(mask)
            Atilde.eliminate_zeros()
            Atilde.sort_indices()

    else:
        # Use computational short-cut for case (ninc == 0) and (nsquare > 0)
        # Calculate Atilde^k only at the sparsity pattern of mask.
        for i in range(nsquare - 1):
            Atilde = Atilde * Atilde

        # Call incomplete mat-mat mult
        AtildeCSC = Atilde.tocsc()
        AtildeCSC.sort_indices()
        mask.sort_indices()
        Atilde.sort_indices()
        amg_core.incomplete_mat_mult_csr(Atilde.indptr, Atilde.indices,
                                         Atilde.data, AtildeCSC.indptr,
                                         AtildeCSC.indices, AtildeCSC.data,
                                         mask.indptr, mask.indices, mask.data,
                                         dimen)

        del AtildeCSC, Atilde
        Atilde = mask
        Atilde.eliminate_zeros()
        Atilde.sort_indices()

    del Dinv, Dinv_A, mask

    # Calculate strength based on constrained min problem of
    # min( z - B*x ), such that
    # (B*x)|_i = z|_i, i.e. they are equal at point i
    # z = (I - (t/k) Dinv A)^k delta_i
    #
    # Strength is defined as the relative point-wise approx. error between
    # B*x and z.  We don't use the full z in this problem, only that part of
    # z that is in the sparsity pattern of A.
    #
    # Can use either the D-norm, and inner product, or l2-norm and inner-prod
    # to solve the constrained min problem.  Using D gives scale invariance.
    #
    # This is a quadratic minimization problem with a linear constraint, so
    # we can build a linear system and solve it to find the critical point,
    # i.e. minimum.
    #
    # We exploit a known shortcut for the case of NullDim = 1.  The shortcut is
    # mathematically equivalent to the longer constrained min. problem

    if NullDim == 1:
        # Use shortcut to solve constrained min problem if B is only a vector
        # Strength(i,j) = | 1 - (z(i)/b(j))/(z(j)/b(i)) |
        # These ratios can be calculated by diagonal row and column scalings

        # Create necessary vectors for scaling Atilde
        #   Its not clear what to do where B == 0.  This is an
        #   an easy programming solution, that may make sense.
        Bmat_forscaling = np.ravel(Bmat)
        Bmat_forscaling[Bmat_forscaling == 0] = 1.0
        DAtilde = Atilde.diagonal()
        DAtildeDivB = np.ravel(DAtilde) / Bmat_forscaling

        # Calculate best approximation, z_tilde, in span(B)
        #   Importantly, scale_rows and scale_columns leave zero entries
        #   in the matrix.  For previous implementations this was useful
        #   because we assume data and Atilde.data are the same length below
        data = Atilde.data.copy()
        Atilde.data[:] = 1.0
        Atilde = scale_rows(Atilde, DAtildeDivB)
        Atilde = scale_columns(Atilde, np.ravel(Bmat_forscaling))

        # If angle in the complex plane between z and z_tilde is
        # greater than 90 degrees, then weak.  We can just look at the
        # dot product to determine if angle is greater than 90 degrees.
        angle = np.real(Atilde.data) * np.real(data) +\
            np.imag(Atilde.data) * np.imag(data)
        angle = angle < 0.0
        angle = np.array(angle, dtype=bool)

        # Calculate Approximation ratio
        Atilde.data = Atilde.data / data

        # If approximation ratio is less than tol, then weak connection
        weak_ratio = (np.abs(Atilde.data) < 1e-4)

        # Calculate Approximation error
        Atilde.data = abs(1.0 - Atilde.data)

        # Set small ratios and large angles to weak
        Atilde.data[weak_ratio] = 0.0
        Atilde.data[angle] = 0.0

        # Set near perfect connections to 1e-4
        Atilde.eliminate_zeros()
        Atilde.data[Atilde.data < np.sqrt(np.finfo(float).eps)] = 1e-4

        del data, weak_ratio, angle

    else:
        # For use in computing local B_i^H*B, precompute the element-wise
        # multiply of each column of B with each other column.  We also scale
        # by 2.0 to account for BDB's eventual use in a constrained
        # minimization problem
        BDBCols = int(np.sum(np.arange(NullDim + 1)))
        BDB = np.zeros((dimen, BDBCols), dtype=A.dtype)
        counter = 0
        for i in range(NullDim):
            for j in range(i, NullDim):
                BDB[:, counter] = 2.0 *\
                    (np.conjugate(np.ravel(np.asarray(B[:, i]))) *
                        np.ravel(np.asarray(D_A * B[:, j])))
                counter = counter + 1

        # Choose tolerance for dropping "numerically zero" values later
        t = Atilde.dtype.char
        eps = np.finfo(np.float).eps
        feps = np.finfo(np.single).eps
        geps = np.finfo(np.longfloat).eps
        _array_precision = {'f': 0, 'd': 1, 'g': 2, 'F': 0, 'D': 1, 'G': 2}
        tol = {0: feps * 1e3, 1: eps * 1e6, 2: geps * 1e6}[_array_precision[t]]

        # Use constrained min problem to define strength
        amg_core.evolution_strength_helper(Atilde.data,
                                           Atilde.indptr,
                                           Atilde.indices,
                                           Atilde.shape[0],
                                           np.ravel(np.asarray(B)),
                                           np.ravel(np.asarray(
                                               (D_A * np.conjugate(B)).T)),
                                           np.ravel(np.asarray(BDB)),
                                           BDBCols, NullDim, tol)

        Atilde.eliminate_zeros()

    # All of the strength values are real by this point, so ditch the complex
    # part
    Atilde.data = np.array(np.real(Atilde.data), dtype=float)

    # Apply drop tolerance
    if epsilon != np.inf:
        amg_core.apply_distance_filter(dimen, epsilon, Atilde.indptr,
                                       Atilde.indices, Atilde.data)
        Atilde.eliminate_zeros()

    # Symmetrize
    if symmetrize_measure:
        Atilde = 0.5 * (Atilde + Atilde.T)

    # Set diagonal to 1.0, as each point is strongly connected to itself.
    Id = sparse.eye(dimen, dimen, format="csr")
    Id.data -= Atilde.diagonal()
    Atilde = Atilde + Id

    # If converted BSR to CSR, convert back and return amalgamated matrix,
    #   i.e. the sparsity structure of the blocks of Atilde
    if not csrflag:
        Atilde = Atilde.tobsr(blocksize=(numPDEs, numPDEs))

        n_blocks = Atilde.indices.shape[0]
        blocksize = Atilde.blocksize[0] * Atilde.blocksize[1]
        CSRdata = np.zeros((n_blocks,))
        amg_core.min_blocks(n_blocks, blocksize,
                            np.ravel(np.asarray(Atilde.data)), CSRdata)
        # Atilde = sparse.csr_matrix((data, row, col), shape=(*,*))
        Atilde = sparse.csr_matrix((CSRdata, Atilde.indices, Atilde.indptr),
                                   shape=(int(Atilde.shape[0] / numPDEs),
                                          int(Atilde.shape[1] / numPDEs)))

    # Standardized strength values require small values be weak and large
    # values be strong.  So, we invert the algebraic distances computed here
    Atilde.data = 1.0 / Atilde.data

    # Scale C by the largest magnitude entry in each row
    Atilde = scale_rows_by_largest_entry(Atilde)

    return Atilde


def relaxation_vectors(A, R, k, alpha):
    """Generate test vectors by relaxing on Ax=0 for some random vectors x.

    Parameters
    ----------
    A : {csr_matrix}
        Sparse NxN matrix
    alpha : scalar
        Weight for Jacobi
    R : integer
        Number of random vectors
    k : integer
        Number of relaxation passes

    Returns
    -------
    x : {array}
        Dense array N x k array of relaxation vectors
    """
    # random n x R block in column ordering
    n = A.shape[0]
    x = np.random.rand(n * R) - 0.5
    x = np.reshape(x, (n, R), order='F')
    # for i in range(R):
    #     x[:,i] = x[:,i] - np.mean(x[:,i])
    b = np.zeros((n, 1))

    for r in range(0, R):
        jacobi(A, x[:, r], b, iterations=k, omega=alpha)
        # x[:,r] = x[:,r]/norm(x[:,r])

    return x


def affinity_distance(A, alpha=0.5, R=5, k=20, epsilon=4.0):
    """Construct an AMG strength of connection matrix using an affinity
    distance measure.

    Parameters
    ----------
    A : {csr_matrix}
        Sparse NxN matrix
    alpha : scalar
        Weight for Jacobi
    R : integer
        Number of random vectors
    k : integer
        Number of relaxation passes
    epsilon : scalar
        Drop tolerance

    Returns
    -------
    C : {csr_matrix}
        Sparse matrix of strength values

    References
    ----------
    .. [LiBr] Oren E. Livne and Achi Brandt, "Lean Algebraic Multigrid
        (LAMG): Fast Graph Laplacian Linear Solver"

    Notes
    -----
    No unit testing yet.

    Does not handle BSR matrices yet.

    See [LiBr]_ for more details.

    """

    if not sparse.isspmatrix_csr(A):
        A = sparse.csr_matrix(A)

    if alpha < 0:
        raise ValueError('expected alpha>0')

    if R <= 0 or not isinstance(R, int):
        raise ValueError('expected integer R>0')

    if k <= 0 or not isinstance(k, int):
        raise ValueError('expected integer k>0')

    if epsilon < 1:
        raise ValueError('expected epsilon>1.0')

    def distance(x):
        (rows, cols) = A.nonzero()
        return 1 - np.sum(x[rows] * x[cols], axis=1)**2 / \
            (np.sum(x[rows]**2, axis=1) * np.sum(x[cols]**2, axis=1))

    return distance_measure_common(A, distance, alpha, R, k, epsilon)


def algebraic_distance(A, alpha=0.5, R=5, k=20, epsilon=2.0, p=2):
    """Construct an AMG strength of connection matrix using an algebraic
    distance measure.

    Parameters
    ----------
    A : {csr_matrix}
        Sparse NxN matrix
    alpha : scalar
        Weight for Jacobi
    R : integer
        Number of random vectors
    k : integer
        Number of relaxation passes
    epsilon : scalar
        Drop tolerance
    p : scalar or inf
        p-norm of the measure

    Returns
    -------
    C : {csr_matrix}
        Sparse matrix of strength values

    References
    ----------
    .. [SaSaSc] Ilya Safro, Peter Sanders, and Christian Schulz,
        "Advanced Coarsening Schemes for Graph Partitioning"

    Notes
    -----
    No unit testing yet.

    Does not handle BSR matrices yet.

    See [SaSaSc]_ for more details.

    """

    if not sparse.isspmatrix_csr(A):
        A = sparse.csr_matrix(A)

    if alpha < 0:
        raise ValueError('expected alpha>0')

    if R <= 0 or not isinstance(R, int):
        raise ValueError('expected integer R>0')

    if k <= 0 or not isinstance(k, int):
        raise ValueError('expected integer k>0')

    if epsilon < 1:
        raise ValueError('expected epsilon>1.0')

    if p < 1:
        raise ValueError('expected p>1 or equal to numpy.inf')

    def distance(x):
        (rows, cols) = A.nonzero()
        if p != np.inf:
            avg = np.sum(np.abs(x[rows] - x[cols])**p, axis=1) / R
            return (avg)**(1.0 / p)
        else:
            return np.abs(x[rows] - x[cols]).max(axis=1)

    return distance_measure_common(A, distance, alpha, R, k, epsilon)


def distance_measure_common(A, func, alpha, R, k, epsilon):
    """
    Helper function to create strength of connection matrix
    from a function applied to relaxation vectors.
    """
    # create test vectors
    x = relaxation_vectors(A, R, k, alpha)

    # apply distance measure function to vectors
    d = func(x)

    # drop distances to self
    (rows, cols) = A.nonzero()
    weak = np.where(rows == cols)[0]
    d[weak] = 0
    C = sparse.csr_matrix((d, (rows, cols)), shape=A.shape)
    C.eliminate_zeros()

    # remove weak connections
    # removes entry e from a row if e > theta * min of all entries in the row
    amg_core.apply_distance_filter(C.shape[0], epsilon, C.indptr,
                                   C.indices, C.data)
    C.eliminate_zeros()

    # Standardized strength values require small values be weak and large
    # values be strong.  So, we invert the distances.
    C.data = 1.0 / C.data

    # Put an identity on the diagonal
    C = C + sparse.eye(C.shape[0], C.shape[1], format='csr')

    # Scale C by the largest magnitude entry in each row
    C = scale_rows_by_largest_entry(C)

    return C
