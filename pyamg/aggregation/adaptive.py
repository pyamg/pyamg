"""Adaptive Smoothed Aggregation"""
from __future__ import absolute_import


from warnings import warn
import numpy as np
import scipy as sp
from scipy.sparse import csr_matrix, bsr_matrix, isspmatrix_csr,\
    isspmatrix_csc, isspmatrix_bsr, eye, SparseEfficiencyWarning

from pyamg.multilevel import multilevel_solver
from pyamg.strength import symmetric_strength_of_connection,\
    classical_strength_of_connection, evolution_strength_of_connection
from ..relaxation.relaxation import gauss_seidel, gauss_seidel_nr,\
    gauss_seidel_ne, gauss_seidel_indexed, jacobi, polynomial
from pyamg.relaxation.smoothing import change_smoothers, rho_D_inv_A
from pyamg.krylov import gmres
from pyamg.util.linalg import norm, approximate_spectral_radius
from .aggregation import smoothed_aggregation_solver
from .aggregate import standard_aggregation, lloyd_aggregation
from .smooth import jacobi_prolongation_smoother,\
    energy_prolongation_smoother, richardson_prolongation_smoother
from .tentative import fit_candidates
from pyamg.util.utils import amalgamate, levelize_strength_or_aggregation, \
    levelize_smooth_or_improve_candidates

__all__ = ['adaptive_sa_solver']


def eliminate_local_candidates(x, AggOp, A, T, Ca=1.0, **kwargs):
    """
    Helper function that determines where to eliminate candidates locally
    on a per aggregate basis.

    Parameters
    ---------
    x : {array}
        n x 1 vector of new candidate
    AggOp : {CSR or CSC sparse matrix}
        Aggregation operator for the level that x was generated for
    A : {sparse matrix}
        Operator for the level that x was generated for
    T : {sparse matrix}
        Tentative prolongation operator for the level that x was generated for
    Ca : {scalar}
        Constant threshold parameter to decide when to drop candidates

    Returns
    -------
    Nothing, x is modified in place
    """

    if not (isspmatrix_csr(AggOp) or isspmatrix_csc(AggOp)):
        raise TypeError('AggOp must be a CSR or CSC matrix')
    else:
        AggOp = AggOp.tocsc()
        ndof = max(x.shape)
        nPDEs = int(ndof/AggOp.shape[0])

    def aggregate_wise_inner_product(z, AggOp, nPDEs, ndof):
        """
        Helper function that calculates <z, z>_i, i.e., the
        inner product of z only over aggregate i
        Returns a vector of length num_aggregates where entry i is <z, z>_i
        """

        z = np.ravel(z)*np.ravel(z)
        innerp = np.zeros((1, AggOp.shape[1]), dtype=z.dtype)
        for j in range(nPDEs):
            innerp += z[slice(j, ndof, nPDEs)].reshape(1, -1) * AggOp

        return innerp.reshape(-1, 1)

    def get_aggregate_weights(AggOp, A, z, nPDEs, ndof):
        """
        Calculate local aggregate quantities
        Return a vector of length num_aggregates where entry i is
        (card(agg_i)/A.shape[0]) ( <Az, z>/rho(A) )
        """
        rho = approximate_spectral_radius(A)
        zAz = np.dot(z.reshape(1, -1), A*z.reshape(-1, 1))
        card = nPDEs*(AggOp.indptr[1:]-AggOp.indptr[:-1])
        weights = (np.ravel(card)*zAz)/(A.shape[0]*rho)
        return weights.reshape(-1, 1)

    # Run test 1, which finds where x is small relative to its energy
    weights = Ca*get_aggregate_weights(AggOp, A, x, nPDEs, ndof)
    mask1 = aggregate_wise_inner_product(x, AggOp, nPDEs, ndof) <= weights

    # Run test 2, which finds where x is already approximated
    # accurately by the existing T
    projected_x = x - T*(T.T*x)
    mask2 = aggregate_wise_inner_product(projected_x,
                                         AggOp, nPDEs, ndof) <= weights

    # Combine masks and zero out corresponding aggregates in x
    mask = np.ravel(mask1 + mask2).nonzero()[0]
    if mask.shape[0] > 0:
        mask = nPDEs*AggOp[:, mask].indices
        for j in range(nPDEs):
            x[mask+j] = 0.0


def unpack_arg(v):
    """Helper function for local methods"""
    if isinstance(v, tuple):
        return v[0], v[1]
    else:
        return v, {}


def adaptive_sa_solver(A, initial_candidates=None, symmetry='hermitian',
                       pdef=True, num_candidates=1, candidate_iters=5,
                       improvement_iters=0, epsilon=0.1,
                       max_levels=10, max_coarse=10, aggregate='standard',
                       prepostsmoother=('gauss_seidel',
                                        {'sweep': 'symmetric'}),
                       smooth=('jacobi', {}), strength='symmetric',
                       coarse_solver='pinv2',
                       eliminate_local=(False, {'Ca': 1.0}), keep=False,
                       **kwargs):
    """
    Create a multilevel solver using Adaptive Smoothed Aggregation (aSA)

    Parameters
    ----------
    A : {csr_matrix, bsr_matrix}
        Square matrix in CSR or BSR format
    initial_candidates : {None, n x m dense matrix}
        If a matrix, then this forms the basis for the first m candidates.
        Also in this case, the initial setup stage is skipped, because this
        provides the first candidate(s).  If None, then a random initial guess
        and relaxation are used to inform the initial candidate.
    symmetry : {string}
        'symmetric' refers to both real and complex symmetric
        'hermitian' refers to both complex Hermitian and real Hermitian
        Note that for the strictly real case, these two options are the same
        Note that this flag does not denote definiteness of the operator
    pdef : {bool}
        True or False, whether A is known to be positive definite.
    num_candidates : {integer} : default 1
        Number of near-nullspace candidates to generate
    candidate_iters : {integer} : default 5
        Number of smoothing passes/multigrid cycles used at each level of
        the adaptive setup phase
    improvement_iters : {integer} : default 0
        Number of times each candidate is improved
    epsilon : {float} : default 0.1
        Target convergence factor
    max_levels : {integer} : default 10
        Maximum number of levels to be used in the multilevel solver.
    max_coarse : {integer} : default 500
        Maximum number of variables permitted on the coarse grid.
    prepostsmoother : {string or dict}
        Pre- and post-smoother used in the adaptive method
    strength : ['symmetric', 'classical', 'evolution', ('predefined', {'C': csr_matrix}), None]
        Method used to determine the strength of connection between unknowns of
        the linear system.  See smoothed_aggregation_solver(...) documentation.
    aggregate : ['standard', 'lloyd', 'naive', ('predefined', {'AggOp': csr_matrix})]
        Method used to aggregate nodes.  See smoothed_aggregation_solver(...)
        documentation.
    smooth : ['jacobi', 'richardson', 'energy', None]
        Method used used to smooth the tentative prolongator.  See
        smoothed_aggregation_solver(...) documentation
    coarse_solver : ['splu', 'lu', 'cholesky, 'pinv', 'gauss_seidel', ... ]
        Solver used at the coarsest level of the MG hierarchy.
        Optionally, may be a tuple (fn, args), where fn is a string such as
        ['splu', 'lu', ...] or a callable function, and args is a dictionary of
        arguments to be passed to fn.
    eliminate_local : {tuple}
        Length 2 tuple.  If the first entry is True, then eliminate candidates
        where they aren't needed locally, using the second entry of the tuple
        to contain arguments to local elimination routine.  Given the rigid
        sparse data structures, this doesn't help much, if at all, with
        complexity.  Its more of a diagnostic utility.
    keep: {bool} : default False
        Flag to indicate keeping extra operators in the hierarchy for
        diagnostics.  For example, if True, then strength of connection (C),
        tentative prolongation (T), and aggregation (AggOp) are kept.

    Returns
    -------
    multilevel_solver : multilevel_solver
        Smoothed aggregation solver with adaptively generated candidates

    Notes
    -----

    - Floating point value representing the "work" required to generate
      the solver.  This value is the total cost of just relaxation, relative
      to the fine grid.  The relaxation method used is assumed to symmetric
      Gauss-Seidel.

    - Unlike the standard Smoothed Aggregation (SA) method, adaptive SA does
      not require knowledge of near-nullspace candidate vectors.  Instead, an
      adaptive procedure computes one or more candidates 'from scratch'.  This
      approach is useful when no candidates are known or the candidates have
      been invalidated due to changes to matrix A.

    Examples
    --------
    >>> from pyamg.gallery import stencil_grid
    >>> from pyamg.aggregation import adaptive_sa_solver
    >>> import numpy as np
    >>> A=stencil_grid([[-1,-1,-1],[-1,8.0,-1],[-1,-1,-1]],\
                       (31,31),format='csr')
    >>> [asa,work] = adaptive_sa_solver(A,num_candidates=1)
    >>> residuals=[]
    >>> x=asa.solve(b=np.ones((A.shape[0],)), x0=np.ones((A.shape[0],)),\
                    residuals=residuals)

    References
    ----------
    .. [1] Brezina, Falgout, MacLachlan, Manteuffel, McCormick, and Ruge
       "Adaptive Smoothed Aggregation ($\alpha$SA) Multigrid"
       SIAM Review Volume 47,  Issue 2  (2005)
       http://www.cs.umn.edu/~maclach/research/aSA2.pdf

    """

    if not (isspmatrix_csr(A) or isspmatrix_bsr(A)):
        try:
            A = csr_matrix(A)
            warn("Implicit conversion of A to CSR", SparseEfficiencyWarning)
        except:
            raise TypeError('Argument A must have type csr_matrix or\
                            bsr_matrix, or be convertible to csr_matrix')

    A = A.asfptype()
    if A.shape[0] != A.shape[1]:
        raise ValueError('expected square matrix')

    # Track work in terms of relaxation
    work = np.zeros((1,))

    # Levelize the user parameters, so that they become lists describing the
    # desired user option on each level.
    max_levels, max_coarse, strength =\
        levelize_strength_or_aggregation(strength, max_levels, max_coarse)
    max_levels, max_coarse, aggregate =\
        levelize_strength_or_aggregation(aggregate, max_levels, max_coarse)
    smooth = levelize_smooth_or_improve_candidates(smooth, max_levels)

    # Develop initial candidate(s).  Note that any predefined aggregation is
    # preserved.
    if initial_candidates is None:
        B, aggregate, strength =\
            initial_setup_stage(A, symmetry, pdef, candidate_iters, epsilon,
                                max_levels, max_coarse, aggregate,
                                prepostsmoother, smooth, strength, work)
        # Normalize B
        B = (1.0/norm(B, 'inf')) * B
        num_candidates -= 1
    else:
        # Otherwise, use predefined candidates
        B = initial_candidates
        num_candidates -= B.shape[1]
        # Generate Aggregation and Strength Operators (the brute force way)
        sa = smoothed_aggregation_solver(A, B=B, symmetry=symmetry,
                                         presmoother=prepostsmoother,
                                         postsmoother=prepostsmoother,
                                         smooth=smooth, strength=strength,
                                         max_levels=max_levels,
                                         max_coarse=max_coarse,
                                         aggregate=aggregate,
                                         coarse_solver=coarse_solver,
                                         improve_candidates=None, keep=True,
                                         **kwargs)
        if len(sa.levels) > 1:
            # Set strength-of-connection and aggregation
            aggregate = [('predefined', {'AggOp': sa.levels[i].AggOp.tocsr()})
                         for i in range(len(sa.levels) - 1)]
            strength = [('predefined', {'C': sa.levels[i].C.tocsr()})
                        for i in range(len(sa.levels) - 1)]

    # Develop additional candidates
    for i in range(num_candidates):
        x = general_setup_stage(
            smoothed_aggregation_solver(A, B=B, symmetry=symmetry,
                                        presmoother=prepostsmoother,
                                        postsmoother=prepostsmoother,
                                        smooth=smooth,
                                        coarse_solver=coarse_solver,
                                        aggregate=aggregate,
                                        strength=strength,
                                        improve_candidates=None,
                                        keep=True, **kwargs),
            symmetry, candidate_iters, prepostsmoother, smooth,
            eliminate_local, coarse_solver, work)

        # Normalize x and add to candidate list
        x = x/norm(x, 'inf')
        if np.isinf(x[0]) or np.isnan(x[0]):
            raise ValueError('Adaptive candidate is all 0.')
        B = np.hstack((B, x.reshape(-1, 1)))

    # Improve candidates
    if B.shape[1] > 1 and improvement_iters > 0:
        b = np.zeros((A.shape[0], 1), dtype=A.dtype)
        for i in range(improvement_iters):
            for j in range(B.shape[1]):
                # Run a V-cycle built on everything except candidate j, while
                # using candidate j as the initial guess
                x0 = B[:, 0]
                B = B[:, 1:]
                sa_temp =\
                    smoothed_aggregation_solver(A, B=B, symmetry=symmetry,
                                                presmoother=prepostsmoother,
                                                postsmoother=prepostsmoother,
                                                smooth=smooth,
                                                coarse_solver=coarse_solver,
                                                aggregate=aggregate,
                                                strength=strength,
                                                improve_candidates=None,
                                                keep=True, **kwargs)
                x = sa_temp.solve(b, x0=x0,
                                  tol=float(np.finfo(np.float).tiny),
                                  maxiter=candidate_iters, cycle='V')
                work[:] += 2 * sa_temp.operator_complexity() *\
                    sa_temp.levels[0].A.nnz * candidate_iters

                # Apply local elimination
                elim, elim_kwargs = unpack_arg(eliminate_local)
                if elim is True:
                    x = x/norm(x, 'inf')
                    eliminate_local_candidates(x, sa_temp.levels[0].AggOp, A,
                                               sa_temp.levels[0].T,
                                               **elim_kwargs)

                # Normalize x and add to candidate list
                x = x/norm(x, 'inf')
                if np.isinf(x[0]) or np.isnan(x[0]):
                    raise ValueError('Adaptive candidate is all 0.')
                B = np.hstack((B, x.reshape(-1, 1)))

    elif improvement_iters > 0:
        # Special case for improving a single candidate
        max_levels = len(aggregate) + 1
        max_coarse = 0
        for i in range(improvement_iters):
            B, aggregate, strength =\
                initial_setup_stage(A, symmetry, pdef, candidate_iters,
                                    epsilon, max_levels, max_coarse,
                                    aggregate, prepostsmoother, smooth,
                                    strength, work, initial_candidate=B)
            # Normalize B
            B = (1.0/norm(B, 'inf'))*B

    return [smoothed_aggregation_solver(A, B=B, symmetry=symmetry,
                                        presmoother=prepostsmoother,
                                        postsmoother=prepostsmoother,
                                        smooth=smooth,
                                        coarse_solver=coarse_solver,
                                        aggregate=aggregate, strength=strength,
                                        improve_candidates=None, keep=keep,
                                        **kwargs),
            work[0]/A.nnz]


def initial_setup_stage(A, symmetry, pdef, candidate_iters, epsilon,
                        max_levels, max_coarse, aggregate, prepostsmoother,
                        smooth, strength, work, initial_candidate=None):
    """
    Computes a complete aggregation and the first near-nullspace candidate
    following Algorithm 3 in Brezina et al.

    Parameters
    ----------
    candidate_iters
        number of test relaxation iterations
    epsilon
        minimum acceptable relaxation convergence factor

    References
    ----------
    .. [1] Brezina, Falgout, MacLachlan, Manteuffel, McCormick, and Ruge
       "Adaptive Smoothed Aggregation ($\alpha$SA) Multigrid"
       SIAM Review Volume 47,  Issue 2  (2005)
       http://www.cs.umn.edu/~maclach/research/aSA2.pdf
    """

    # Define relaxation routine
    def relax(A, x):
        fn, kwargs = unpack_arg(prepostsmoother)
        if fn == 'gauss_seidel':
            gauss_seidel(A, x, np.zeros_like(x),
                         iterations=candidate_iters, sweep='symmetric')
        elif fn == 'gauss_seidel_nr':
            gauss_seidel_nr(A, x, np.zeros_like(x),
                            iterations=candidate_iters, sweep='symmetric')
        elif fn == 'gauss_seidel_ne':
            gauss_seidel_ne(A, x, np.zeros_like(x),
                            iterations=candidate_iters, sweep='symmetric')
        elif fn == 'jacobi':
            jacobi(A, x, np.zeros_like(x), iterations=1,
                   omega=1.0 / rho_D_inv_A(A))
        elif fn == 'richardson':
            polynomial(A, x, np.zeros_like(x), iterations=1,
                       coefficients=[1.0/approximate_spectral_radius(A)])
        elif fn == 'gmres':
            x[:] = (gmres(A, np.zeros_like(x), x0=x,
                    maxiter=candidate_iters)[0]).reshape(x.shape)
        else:
            raise TypeError('Unrecognized smoother')

    # flag for skipping steps f-i in step 4
    skip_f_to_i = True

    # step 1
    A_l = A
    if initial_candidate is None:
        x = sp.rand(A_l.shape[0], 1).astype(A_l.dtype)
        # The following type check matches the usual 'complex' type,
        # but also numpy data types such as 'complex64', 'complex128'
        # and 'complex256'.
        if A_l.dtype.name.startswith('complex'):
            x = x + 1.0j*sp.rand(A_l.shape[0], 1)
    else:
        x = np.array(initial_candidate, dtype=A_l.dtype)

    # step 2
    relax(A_l, x)
    work[:] += A_l.nnz * candidate_iters*2

    # step 3
    # not advised to stop the iteration here: often the first relaxation pass
    # _is_ good, but the remaining passes are poor
    # if x_A_x/x_A_x_old < epsilon:
    #    # relaxation alone is sufficient
    #    print 'relaxation alone works: %g'%(x_A_x/x_A_x_old)
    #    return x, []

    # step 4
    As = [A]
    xs = [x]
    Ps = []
    AggOps = []
    StrengthOps = []

    while A.shape[0] > max_coarse and max_levels > 1:
        # The real check to break from the while loop is below

        # Begin constructing next level
        fn, kwargs = unpack_arg(strength[len(As)-1])  # step 4b
        if fn == 'symmetric':
            C_l = symmetric_strength_of_connection(A_l, **kwargs)
            # Diagonal must be nonzero
            C_l = C_l + eye(C_l.shape[0], C_l.shape[1], format='csr')
        elif fn == 'classical':
            C_l = classical_strength_of_connection(A_l, **kwargs)
            # Diagonal must be nonzero
            C_l = C_l + eye(C_l.shape[0], C_l.shape[1], format='csr')
            if isspmatrix_bsr(A_l):
                C_l = amalgamate(C_l, A_l.blocksize[0])
        elif (fn == 'ode') or (fn == 'evolution'):
            C_l = evolution_strength_of_connection(A_l,
                                                   np.ones(
                                                       (A_l.shape[0], 1),
                                                       dtype=A.dtype),
                                                   **kwargs)
        elif fn == 'predefined':
            C_l = kwargs['C'].tocsr()
        elif fn is None:
            C_l = A_l.tocsr()
        else:
            raise ValueError('unrecognized strength of connection method: %s' %
                             str(fn))

        # In SA, strength represents "distance", so we take magnitude of
        # complex values
        if C_l.dtype.name.startswith('complex'):
            C_l.data = np.abs(C_l.data)

        # Create a unified strength framework so that large values represent
        # strong connections and small values represent weak connections
        if (fn == 'ode') or (fn == 'evolution') or (fn == 'energy_based'):
            C_l.data = 1.0 / C_l.data

        # aggregation
        fn, kwargs = unpack_arg(aggregate[len(As) - 1])
        if fn == 'standard':
            AggOp = standard_aggregation(C_l, **kwargs)[0]
        elif fn == 'lloyd':
            AggOp = lloyd_aggregation(C_l, **kwargs)[0]
        elif fn == 'predefined':
            AggOp = kwargs['AggOp'].tocsr()
        else:
            raise ValueError('unrecognized aggregation method %s' % str(fn))

        T_l, x = fit_candidates(AggOp, x)  # step 4c

        fn, kwargs = unpack_arg(smooth[len(As)-1])  # step 4d
        if fn == 'jacobi':
            P_l = jacobi_prolongation_smoother(A_l, T_l, C_l, x, **kwargs)
        elif fn == 'richardson':
            P_l = richardson_prolongation_smoother(A_l, T_l, **kwargs)
        elif fn == 'energy':
            P_l = energy_prolongation_smoother(A_l, T_l, C_l, x, None,
                                               (False, {}), **kwargs)
        elif fn is None:
            P_l = T_l
        else:
            raise ValueError('unrecognized prolongation smoother method %s' %
                             str(fn))

        # R should reflect A's structure # step 4e
        if symmetry == 'symmetric':
            A_l = P_l.T.asformat(P_l.format) * A_l * P_l
        elif symmetry == 'hermitian':
            A_l = P_l.H.asformat(P_l.format) * A_l * P_l

        StrengthOps.append(C_l)
        AggOps.append(AggOp)
        Ps.append(P_l)
        As.append(A_l)

        # skip to step 5 as in step 4e
        if (A_l.shape[0] <= max_coarse) or (len(AggOps) + 1 >= max_levels):
            break

        if not skip_f_to_i:
            x_hat = x.copy()  # step 4g
            relax(A_l, x)  # step 4h
            work[:] += A_l.nnz*candidate_iters*2
            if pdef is True:
                x_A_x = np.dot(np.conjugate(x).T, A_l*x)
                xhat_A_xhat = np.dot(np.conjugate(x_hat).T, A_l*x_hat)
                err_ratio = (x_A_x/xhat_A_xhat)**(1.0/candidate_iters)
            else:
                # use A.H A inner-product
                Ax = A_l * x
                # Axhat = A_l * x_hat
                x_A_x = np.dot(np.conjugate(Ax).T, Ax)
                xhat_A_xhat = np.dot(np.conjugate(x_hat).T, A_l*x_hat)
                err_ratio = (x_A_x/xhat_A_xhat)**(1.0/candidate_iters)

            if err_ratio < epsilon:  # step 4i
                # print "sufficient convergence, skipping"
                skip_f_to_i = True
                if x_A_x == 0:
                    x = x_hat  # need to restore x
        else:
            # just carry out relaxation, don't check for convergence
            relax(A_l, x)  # step 4h
            work[:] += 2 * A_l.nnz * candidate_iters

        # store xs for diagnostic use and for use in step 5
        xs.append(x)

    # step 5
    # Extend coarse-level candidate to the finest level
    # --> note that we start with the x from the second coarsest level
    x = xs[-1]
    # make sure that xs[-1] has been relaxed by step 4h, i.e. relax(As[-2], x)
    for lev in range(len(Ps)-2, -1, -1):  # lev = coarsest ... finest-1
        P = Ps[lev]                     # I: lev --> lev+1
        A = As[lev]                     # A on lev+1
        x = P * x
        relax(A, x)
        work[:] += A.nnz*candidate_iters*2

    # Set predefined strength of connection and aggregation
    if len(AggOps) > 1:
        aggregate = [('predefined', {'AggOp': AggOps[i]})
                     for i in range(len(AggOps))]
        strength = [('predefined', {'C': StrengthOps[i]})
                    for i in range(len(StrengthOps))]

    return x, aggregate, strength  # first candidate


def general_setup_stage(ml, symmetry, candidate_iters, prepostsmoother,
                        smooth, eliminate_local, coarse_solver, work):
    """
    Computes additional candidates and improvements
    following Algorithm 4 in Brezina et al.

    Parameters
    ----------
    candidate_iters
        number of test relaxation iterations
    epsilon
        minimum acceptable relaxation convergence factor

    References
    ----------
    .. [1] Brezina, Falgout, MacLachlan, Manteuffel, McCormick, and Ruge
       "Adaptive Smoothed Aggregation (alphaSA) Multigrid"
       SIAM Review Volume 47,  Issue 2  (2005)
       http://www.cs.umn.edu/~maclach/research/aSA2.pdf
    """

    def make_bridge(T):
        M, N = T.shape
        K = T.blocksize[0]
        bnnz = T.indptr[-1]
        # the K+1 represents the new dof introduced by the new candidate.  the
        # bridge 'T' ignores this new dof and just maps zeros there
        data = np.zeros((bnnz, K+1, K), dtype=T.dtype)
        data[:, :-1, :] = T.data
        return bsr_matrix((data, T.indices, T.indptr),
                          shape=((K + 1) * int(M / K), N))

    def expand_candidates(B_old, nodesize):
        # insert a new dof that is always zero, to create NullDim+1 dofs per
        # node in B
        NullDim = B_old.shape[1]
        nnodes = int(B_old.shape[0] / nodesize)
        Bnew = np.zeros((nnodes, nodesize+1, NullDim), dtype=B_old.dtype)
        Bnew[:, :-1, :] = B_old.reshape(nnodes, nodesize, NullDim)
        return Bnew.reshape(-1, NullDim)

    levels = ml.levels

    x = sp.rand(levels[0].A.shape[0], 1)
    if levels[0].A.dtype.name.startswith('complex'):
        x = x + 1.0j*sp.rand(levels[0].A.shape[0], 1)
    b = np.zeros_like(x)

    x = ml.solve(b, x0=x, tol=float(np.finfo(np.float).tiny),
                 maxiter=candidate_iters)
    work[:] += ml.operator_complexity()*ml.levels[0].A.nnz*candidate_iters*2

    T0 = levels[0].T.copy()

    # TEST FOR CONVERGENCE HERE

    for i in range(len(ml.levels) - 2):
        # alpha-SA paper does local elimination here, but after talking
        # to Marian, its not clear that this helps things
        # fn, kwargs = unpack_arg(eliminate_local)
        # if fn == True:
        #    eliminate_local_candidates(x,levels[i].AggOp,levels[i].A,
        #    levels[i].T, **kwargs)

        # add candidate to B
        B = np.hstack((levels[i].B, x.reshape(-1, 1)))

        # construct Ptent
        T, R = fit_candidates(levels[i].AggOp, B)

        levels[i].T = T
        x = R[:, -1].reshape(-1, 1)

        # smooth P
        fn, kwargs = unpack_arg(smooth[i])
        if fn == 'jacobi':
            levels[i].P = jacobi_prolongation_smoother(levels[i].A, T,
                                                       levels[i].C, R,
                                                       **kwargs)
        elif fn == 'richardson':
            levels[i].P = richardson_prolongation_smoother(levels[i].A, T,
                                                           **kwargs)
        elif fn == 'energy':
            levels[i].P = energy_prolongation_smoother(levels[i].A, T,
                                                       levels[i].C, R, None,
                                                       (False, {}), **kwargs)
            x = R[:, -1].reshape(-1, 1)
        elif fn is None:
            levels[i].P = T
        else:
            raise ValueError('unrecognized prolongation smoother method %s' %
                             str(fn))

        # construct R
        if symmetry == 'symmetric':  # R should reflect A's structure
            levels[i].R = levels[i].P.T.asformat(levels[i].P.format)
        elif symmetry == 'hermitian':
            levels[i].R = levels[i].P.H.asformat(levels[i].P.format)

        # construct coarse A
        levels[i+1].A = levels[i].R * levels[i].A * levels[i].P

        # construct bridging P
        T_bridge = make_bridge(levels[i+1].T)
        R_bridge = levels[i+2].B

        # smooth bridging P
        fn, kwargs = unpack_arg(smooth[i+1])
        if fn == 'jacobi':
            levels[i+1].P = jacobi_prolongation_smoother(levels[i+1].A,
                                                         T_bridge,
                                                         levels[i+1].C,
                                                         R_bridge, **kwargs)
        elif fn == 'richardson':
            levels[i+1].P = richardson_prolongation_smoother(levels[i+1].A,
                                                             T_bridge,
                                                             **kwargs)
        elif fn == 'energy':
            levels[i+1].P = energy_prolongation_smoother(levels[i+1].A,
                                                         T_bridge,
                                                         levels[i+1].C,
                                                         R_bridge, None,
                                                         (False, {}), **kwargs)
        elif fn is None:
            levels[i+1].P = T_bridge
        else:
            raise ValueError('unrecognized prolongation smoother method %s' %
                             str(fn))

        # construct the "bridging" R
        if symmetry == 'symmetric':  # R should reflect A's structure
            levels[i+1].R = levels[i+1].P.T.asformat(levels[i+1].P.format)
        elif symmetry == 'hermitian':
            levels[i+1].R = levels[i+1].P.H.asformat(levels[i+1].P.format)

        # run solver on candidate
        solver = multilevel_solver(levels[i+1:], coarse_solver=coarse_solver)
        change_smoothers(solver, presmoother=prepostsmoother,
                         postsmoother=prepostsmoother)
        x = solver.solve(np.zeros_like(x), x0=x,
                         tol=float(np.finfo(np.float).tiny),
                         maxiter=candidate_iters)
        work[:] += 2 * solver.operator_complexity() * solver.levels[0].A.nnz *\
            candidate_iters*2

        # update values on next level
        levels[i+1].B = R[:, :-1].copy()
        levels[i+1].T = T_bridge

    # note that we only use the x from the second coarsest level
    fn, kwargs = unpack_arg(prepostsmoother)
    for lvl in reversed(levels[:-2]):
        x = lvl.P * x
        work[:] += lvl.A.nnz*candidate_iters*2

        if fn == 'gauss_seidel':
            # only relax at nonzeros, so as not to mess up any locally dropped
            # candidates
            indices = np.ravel(x).nonzero()[0]
            gauss_seidel_indexed(lvl.A, x, np.zeros_like(x), indices,
                                 iterations=candidate_iters, sweep='symmetric')

        elif fn == 'gauss_seidel_ne':
            gauss_seidel_ne(lvl.A, x, np.zeros_like(x),
                            iterations=candidate_iters, sweep='symmetric')

        elif fn == 'gauss_seidel_nr':
            gauss_seidel_nr(lvl.A, x, np.zeros_like(x),
                            iterations=candidate_iters, sweep='symmetric')

        elif fn == 'jacobi':
            jacobi(lvl.A, x, np.zeros_like(x), iterations=1,
                   omega=1.0 / rho_D_inv_A(lvl.A))

        elif fn == 'richardson':
            polynomial(lvl.A, x, np.zeros_like(x), iterations=1,
                       coefficients=[1.0/approximate_spectral_radius(lvl.A)])

        elif fn == 'gmres':
            x[:] = (gmres(lvl.A, np.zeros_like(x), x0=x,
                          maxiter=candidate_iters)[0]).reshape(x.shape)
        else:
            raise TypeError('Unrecognized smoother')

    # x will be dense again, so we have to drop locally again
    elim, elim_kwargs = unpack_arg(eliminate_local)
    if elim is True:
        x = x/norm(x, 'inf')
        eliminate_local_candidates(x, levels[0].AggOp, levels[0].A, T0,
                                   **elim_kwargs)

    return x.reshape(-1, 1)
