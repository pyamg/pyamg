"""Adaptive Bootstrap AMG."""

import numpy as np
import scipy as sp
from scipy.sparse import csr_matrix, isspmatrix_csr, isspmatrix_bsr, eye
from scipy.sparse.linalg.interface import LinearOperator

from ..multilevel import MultilevelSolver
from ..strength import symmetric_strength_of_connection, evolution_strength_of_connection
from ..relaxation.relaxation import gauss_seidel, gauss_seidel_nr, gauss_seidel_ne
from ..relaxation.smoothing import change_smoothers
from ..util.utils import get_Cpt_params, scale_T
from ..util.linalg import norm

from .aggregate import standard_aggregation, naive_aggregation
from .smooth import energy_prolongation_smoother
from .tentative import fit_candidates


def blocksize(A):
    """Return the blocksize of a matrix."""
    if isspmatrix_bsr(A):
        return A.blocksize[0]
    return 1


def unpack_arg(v):
    """Unpack arguments for local methods."""
    if isinstance(v, tuple):
        return v[0], v[1]
    return v, {}


def operator_inner_product(x, y, A):
    """Compute <A x, y>."""
    x = np.ravel(x)
    y = np.ravel(y)
    return np.dot((A*x).conj(), y)


def scale_candidates(B, Bc, A):
    """Scale B_i and Bc_i by the energy norm of column B_i."""
    for j in range(B.shape[1]):
        BdotB = np.dot(B[:, j].conj(), B[:, j])
        alpha = operator_inner_product(B[:, j], B[:, j], A) / BdotB
        B[:, j] /= alpha
        Bc[:, j] /= alpha

    return B, Bc


def cost(smoother):
    """Determine the cost multiplier for the relaxation method in smoother.

    For example, the symmetric option for gauss_seidel results in a multiplier
    that is a factor of 2, and number of iterations is also a multiplier.
    """
    _, kwargs = unpack_arg(smoother)
    multiplier = 1.0
    if 'iterations' in kwargs:
        multiplier *= kwargs['iterations']

    if 'sweep' in kwargs:
        if kwargs['sweep'] == 'symmetric':
            multiplier *= 2.0

    return multiplier


def eigen_relaxation_as_linear_operator(A, M, smoother, candidate_iters):
    """Return LinearOperator object that applies relaxation based on A - <Ax, x>/<Mx, x> M.

    This assumes a homogeneous right-hand-side.

    Parameters
    ----------
    A : {csr_matrix, bsr_matrix}
        Square matrix in CSR or BSR format
    M : {csr_matrix, bsr_matrix}
        Square matrix in CSR or BSR format
    smoother : {tuple, string} :
        Defines the smoother.  If string, use values like 'gauss_seidel'.
        If tuple, use values like ('gauss_seidel', {'sweep':'symmetric'}),
        where the first tuple value is the relaxation method, and the second
        tuple value is a parameter dictionary.
    candidate_iters : {int}
        Number of smoothing iterations

    Returns
    -------
    LinearOperator object that can be used to "multiply" vectors to carry out
    relaxation based on the generalized eigenproblem
    Ax = lambda M x.
    The relaxation is equivalent to relaxing with the matrix
    A - <Ax, x>/<Mx, x> M
    for a homogeneous right-hand-side.
    """
    fn, kwargs = unpack_arg(smoother)
    b = np.zeros((A.shape[0],), dtype=A.dtype)

    if fn == 'gauss_seidel':
        def matvec(x):
            x = x.T.copy()
            for j in range(x.shape[0]):
                for _ in range(candidate_iters):
                    alpha = 0.0
                    numer = operator_inner_product(x[j, :], x[j, :], A)
                    denom = operator_inner_product(x[j, :], x[j, :], M)
                    if denom != 0.0:
                        alpha = numer/denom
                    gauss_seidel(A - alpha*M, x[j, :], b, **kwargs)
            return x.T.copy()
    elif fn == 'gauss_seidel_ne':
        def matvec(x):
            x = x.T.copy()
            for j in range(x.shape[0]):
                for _ in range(candidate_iters):
                    alpha = 0.0
                    numer = operator_inner_product(x[j, :], x[j, :], A)
                    denom = operator_inner_product(x[j, :], x[j, :], M)
                    if denom != 0.0:
                        alpha = numer/denom
                    gauss_seidel_ne(A - alpha*M, x[j, :], b, **kwargs)
            return x.T.copy()
    elif fn == 'gauss_seidel_nr':
        def matvec(x):
            x = x.T.copy()
            for j in range(x.shape[0]):
                for _ in range(candidate_iters):
                    alpha = 0.0
                    numer = operator_inner_product(x[j, :], x[j, :], A)
                    denom = operator_inner_product(x[j, :], x[j, :], M)
                    if denom != 0.0:
                        alpha = numer/denom
                    gauss_seidel_nr(A - alpha*M, x[j, :], b, **kwargs)
            return x.T.copy()
    #  elif: other relaxations possible: CG, GMRES, polynomial and schwarz
    else:
        raise TypeError('Unrecognized smoother')

    return LinearOperator(A.shape, matvec, dtype=A.dtype)


def relaxation_as_linear_operator(A, smoother, candidate_iters):
    """Return LinearOperator object that applies relaxation based on A.

    This assumes a homogeneous right-hand-side.

    Parameters
    ----------
    A : {csr_matrix, bsr_matrix}
        Square matrix in CSR or BSR format
    smoother : {tuple, string} :
        Defines the smoother.  If string, use values like 'gauss_seidel'.
        If tuple, use values like ('gauss_seidel', {'sweep':'symmetric'}),
        where the first tuple value is the relaxation method, and the second
        tuple value is a parameter dictionary.
    candidate_iters : {int}
        Number of smoothing iterations

    Returns
    -------
    LinearOperator object that can be used to "multiply" vectors to carry out
    relaxation based on the matrix A for a homogeneous right-hand-side.
    """
    fn, kwargs = unpack_arg(smoother)
    b = np.zeros((A.shape[0],), dtype=A.dtype)

    if fn == 'gauss_seidel':
        def matvec(x):
            x = x.T.copy()
            for j in range(x.shape[0]):
                for _ in range(candidate_iters):
                    gauss_seidel(A, x[j, :], b, **kwargs)
            return x.T.copy()
    elif fn == 'gauss_seidel_ne':
        def matvec(x):
            x = x.T.copy()
            for j in range(x.shape[0]):
                for _ in range(candidate_iters):
                    gauss_seidel_ne(A, x[j, :], b, **kwargs)
            return x.T.copy()
    elif fn == 'gauss_seidel_nr':
        def matvec(x):
            x = x.T.copy()
            for j in range(x.shape[0]):
                for _ in range(candidate_iters):
                    gauss_seidel_nr(A, x[j, :], b, **kwargs)
            return x.T.copy()
    # elif: other relaxations possible: CG, GMRES, polynomial and schwarz
    else:
        raise TypeError('Unrecognized smoother')

    return LinearOperator(A.shape, matvec, dtype=A.dtype)


def bootstrap_solver(A,
                     symmetry='hermitian',
                     initial_candidates=None,
                     cands_to_freeze=None,
                     num_candidates=3,
                     num_eigvects=3,
                     candidate_iters=5,
                     outer_iters=1,
                     mu=1,
                     max_levels=10,
                     max_coarse=100,
                     strength='symmetric',
                     aggregate='standard',
                     reaggregate=False,
                     smooth=('energy', {'krylov': 'cg', 'maxiter': 4,
                                        'degree': 2, 'weighting': 'local'}),
                     coarse_solver='pinv2',
                     prepostsmoother=('gauss_seidel', {'sweep': 'symmetric'}),
                     verbose=False):
    """Construct a bootstrap AMG solver in a rootnode SA framework.

    Parameters
    ----------
    A : {csr_matrix, bsr_matrix}
        Square matrix in CSR or BSR format
    symmetry : {string}
        -- Not supported --
    initial_candidates : {array}
        Initial guess for the near nullspace modes used to construct the
        solver.  If A is (n x n), then this array is (n x k), where k is the
        number of initial candidate vectors.
    cands_to_freeze : {None, list}
        List of candidates (if any) to freeze and never update during the adaptive
        cycling.  For instance, cands_to_freeze=[0] would be sensible if the first
        initial_candidate were the constant vector for a diffusion problem.
    num_candidates : {integer}
        Number of near-nullspace candidates to generate
    num_eigvects : {integer}
        Number of near-nullspace eigenvectors to generate
        Note: both the candidates and eigenvectors are used to construct interpolation
    candidate_iters : {integer}
        Number of smoothing passes used at each level of the adaptive setup
    outer_iters : {integer}
        Number of outer iterations used in the adaptive setup, i.e., the number
        of times the bootstrap process is called
    mu : {integer}
        Determines the structure of the adaptive cycling.  mu=1 uses V-cycles during
        the adaptive setup, mu=2 uses W-cycles, and so on.
    max_levels : {integer}
        Maximum number of levels to be used in the multilevel solver
    max_coarse : {integer}
        Maximum number of variables permitted on the coarse grid
    strength : ['symmetric', 'classical',
                'evolution', ('predefined', {'C' : csr_matrix}), None]
        Method used to determine the strength of connection between unknowns
        of the linear system.
    aggregate : ['standard', 'naive', ('predefined', {'AggOp' : csr_matrix})]
        Method used to aggregate nodes.  See rootnode_solver(...) documentation.
    reaggregate : {bool}
        Boolean variable that toggles recomputation of strength-of-connection and
        aggregation each time a level is reached during the adaptive cycling.
    smooth : ['energy', None]
        Method used used to smooth the tentative prolongator.
    coarse_solver : ['splu', 'lu', 'cholesky, 'pinv', 'gauss_seidel', ... ]
        Solver used at the coarsest level of the MG hierarchy.
            Optionally, may be a tuple (fn, args), where fn is a string such as
        ['splu', 'lu', ...] or a callable function, and args is a dictionary of
        arguments to be passed to fn.
    prepostsmoother : {string or dict}
        Pre- and post-smoother used in the adaptive method.  If string, use
        values like 'gauss_seidel'.  If tuple, use values like ('gauss_seidel',
        {'sweep':'symmetric'}), where the first tuple value is the relaxation
        method, and the second tuple value is a parameter dictionary.
    verbose : bool
        Print diagnostic statements in bootstrap_setup

    Returns
    -------
    ml : MultilevelSolver
        Rootnode-style smoothed aggregation solver with adaptively generated candidates

    work : {float}
        Value representing the "work" in FLOPS, relative to one fine grid
        relaxation sweep, required to generate the solver.

    Notes
    -----
    - Unlike the standard Smoothed Aggregation (SA) or rootnode method, this
      adaptive solver does not require knowledge of near-nullspace candidate
      vectors.  Instead, an adaptive procedure computes one or more candidates
      'from scratch'.  This approach is useful when no candidates are known or
      the candidates have been invalidated due to changes to matrix A.

    - Both the adaptive candidates and eigenvectors are used to construct
      interpolation

    Examples
    --------
    >>> from pyamg.gallery import stencil_grid
    >>> from pyamg import bootstrap_solver
    >>> import numpy as np
    >>> A=stencil_grid([[-1,-1,-1],[-1,8.0,-1],[-1,-1,-1]], (31,31),format='csr')
    >>> [asa,work] = bootstrap_solver(A,num_candidates=1)
    >>> residuals=[]
    >>> x=asa.solve(b=np.ones((A.shape[0],)),x0=np.ones((A.shape[0],)),residuals=residuals)

    References
    ----------
    .. [1] BAMG reference from Karsten, Brannick, Achi, Ira, et al.
    """
    # Check format of input matrix A
    if not (isspmatrix_csr(A) or isspmatrix_bsr(A)):
        A = csr_matrix(A)
        print('Implicit conversion of A to CSR in pyamg.adaptive_rootnode_solver')

    A = A.asfptype()
    if A.shape[0] != A.shape[1]:
        raise ValueError('expected square matrix')

    # Track work in terms of relaxation
    work = np.zeros((1,))

    # Construct initial candidates, using a random initial guess for
    # all candidates not set a priori by user with initial_candidates
    if initial_candidates is None:
        U = np.zeros((A.shape[0], 0), dtype=A.dtype)
    else:
        U = initial_candidates
        if len(U.shape) == 1:
            U = U.reshape(-1, 1)
        if U.shape[0] != A.shape[0]:
            raise ValueError('Initial candidates and A must have compatible dimensions')

    num_new_cands = num_candidates - U.shape[1]
    if num_new_cands > 0:
        U = np.hstack((U, np.random.rand(U.shape[0], num_new_cands)))
        if A.dtype == complex:
            U[:, -num_new_cands:] += 1.0j * np.random.rand(A.shape[0], num_new_cands)

    # Compute indices of candidates to "freeze" and not adaptively updated
    if cands_to_freeze is None:
        not_frozen = np.arange(U.shape[1])
    else:
        frozen = np.array(cands_to_freeze, dtype=int)
        not_frozen = np.setdiff1d(np.arange(U.shape[1]), frozen)
        if (frozen.shape[0] > 0) and (frozen.min() < 0 or frozen.max() > (U.shape[1]-1)):
            raise ValueError('Candidates to freeze contains invalid indices')

    # Setup initial solver structure
    # M: algebraic mass matrix
    # V: eigenmode estimates at each level
    # U: new candidates are stored as U at each
    #    level in the hierarchy.
    levels = []
    levels.append(MultilevelSolver.Level())
    levels[-1].A = A
    levels[-1].V = np.zeros((A.shape[0], 0), dtype=A.dtype)
    levels[-1].U = U
    levels[-1].M = eye(A.shape[0], A.shape[1], format='csr', dtype=A.dtype)

    # Run bootstrap setup
    for _ in range(outer_iters):
        bootstrap_setup(levels, 0, aggregate, max_coarse, max_levels, strength,
                      prepostsmoother, smooth, coarse_solver, work, symmetry,
                      mu, candidate_iters, not_frozen, reaggregate, num_eigvects, verbose)

    # Return
    sa = MultilevelSolver(levels, coarse_solver=coarse_solver)
    change_smoothers(sa, presmoother=prepostsmoother, postsmoother=prepostsmoother)
    return sa, work[0]/A.nnz


def bootstrap_setup(levels, l, aggregate, max_coarse, max_levels, strength,
                  prepostsmoother, smooth, coarse_solver, work, symmetry,
                  mu, candidate_iters, not_frozen, recoarsen, num_eigvects, verbose):
    """Compute additional candidates/levels, recursively using bootstrap.

    Parameters
    ----------
    levels : {list}
        List of hierarchy levels to adaptive update.  At least the
        initial level (level 0) must be initialized with values for
        A, M, U and V.
    l : {int}
        Integer value of level where adaptive cycling is to start.

    For further parameter information, see above parameter descriptions for
    bootstrap_solver(...).


    Returns
    -------
    levels is modified in place, reflecting the bootstrap process
    """
    if verbose:
        print(f'\nStarting level {l}')
    lvl = levels[l]
    A = lvl.A
    M = lvl.M

    if len(levels) == max_levels or A.shape[0] <= max_coarse:
        # Reached coarsest level

        # Coarsenings can change -- so truncate any obsolete coarser levels
        for _ in range(len(levels) - l - 1):
            levels.pop()

        # Do eigensolve and return smallest eigenmodes in V
        [E, V] = sp.linalg.eig(A.todense(), M.todense())
        indices = np.argsort(np.abs(E))
        if symmetry == 'hermitian':
            lvl.V = np.real(V[:, indices[:num_eigvects]])
        else:
            lvl.V = V[:, indices[:num_eigvects]]

        if verbose:
            print(f'\nFinished coarsest level {l}')

    else:
        # Not at coarsest level, carry out recursive call

        # Presmooth and normalize current candidates, U
        if len(not_frozen) > 0:
            work[0] += A.nnz*candidate_iters*len(not_frozen)*cost(prepostsmoother)
            G = relaxation_as_linear_operator(A, prepostsmoother, candidate_iters)
            lvl.U[:, not_frozen] = G*lvl.U[:, not_frozen]
            for j in not_frozen:
                alpha = norm(lvl.U[:, j], 'inf')
                if alpha != 0.0:
                    lvl.U[:, j] /= norm(lvl.U[:, j], 'inf')

        # Presmooth and normalize current eigenvectors, V
        # This step skipped during 1st downward pass, because V is uninitialized
        if lvl.V.shape[1] > 0:
            work[0] += A.nnz*candidate_iters*lvl.V.shape[1]*cost(prepostsmoother)
            G = eigen_relaxation_as_linear_operator(A, M, prepostsmoother, candidate_iters)
            lvl.V = G*lvl.V
            for j in range(lvl.V.shape[1]):
                alpha = norm(lvl.V[:, j], 'inf')
                if alpha != 0.0:
                    lvl.V[:, j] /= alpha

        # Carry out mu recursive calls to bootstrap process
        for mu_i in range(mu):

            # Decide if strength-of-connection and aggregation need to computed
            compute_aggregation = (not hasattr(lvl, 'T')) or (recoarsen and (mu_i == 0))

            # Strength
            if compute_aggregation:
                if isinstance(strength, list):
                    fn, kwargs = unpack_arg(strength[len(levels)-1])
                else:
                    fn, kwargs = unpack_arg(strength)
                if fn == 'symmetric':
                    C = symmetric_strength_of_connection(A, **kwargs)
                    C = C + eye(C.shape[0], C.shape[1], format='csr')  # nonzero diagonal
                elif fn in ['ode', 'evolution']:
                    B = np.hstack((lvl.U, lvl.V))
                    if 'B' in kwargs:
                        C = evolution_strength_of_connection(A, **kwargs)
                    else:
                        C = evolution_strength_of_connection(A, B, **kwargs)
                    work[0] += A.nnz
                elif fn == 'predefined':
                    C = kwargs['C'].tocsr()
                elif fn is None:
                    C = A.tocsr().copy()
                else:
                    raise ValueError(f'unrecognized strength of connection: {str(fn)}')

                # Strength represents "distance", so we take magnitude of complex values
                if C.dtype == complex:
                    C.data = np.abs(C.data)

                # Create a unified strength framework, where large values represent
                # strong connections and small values represent weak connections
                if fn in ['ode', 'evolution']:
                    C.data = 1.0/C.data

            else:
                ##
                # Use old strength matrix
                C = lvl.C
                Cpts = lvl.Cpts

            # Aggregate
            if compute_aggregation:
                if isinstance(aggregate, list):
                    fn, kwargs = unpack_arg(aggregate[len(levels)-1])
                else:
                    fn, kwargs = unpack_arg(aggregate)
                if fn == 'standard':
                    AggOp, Cnodes = standard_aggregation(C, **kwargs)
                elif fn == 'naive':
                    AggOp, Cnodes = naive_aggregation(C, **kwargs)
                elif fn == 'predefined':
                    AggOp = kwargs['AggOp'].tocsr()
                    Cnodes = kwargs['Cnodes']
                else:
                    raise ValueError(f'unrecognized aggregation method {str(fn)}')
            else:
                # Use old aggregation operator
                AggOp = lvl.AggOp
                Cnodes = lvl.Cnodes

            # Tentative Prolongation
            if compute_aggregation:
                T, dummy_variable = fit_candidates(AggOp,
                                                   np.ones((A.shape[0], 1), dtype=A.dtype))
                Cpt_params = get_Cpt_params(A, Cnodes, AggOp, T)
                I_C = Cpt_params['I_C']     # I_C zeros out all F-pts
                I_F = Cpt_params['I_F']     # I_F zeros out all C-pts
                P_I = Cpt_params['P_I']     # P_I is strict injection
                Cpts = Cpt_params['Cpts']   # C-points (as opposed to nodes)
                T = scale_T(T, P_I, I_F)    # Scale T to inject from coarse-grid
            else:
                # Use old tentative prolongation
                T = lvl.T
                I_C = lvl.I_C
                I_F = lvl.I_F
                P_I = lvl.P_I
                Cpts = lvl.Cpts

            # Inject U and V to the coarse grid
            Uc = P_I.T*lvl.U
            Vc = P_I.T*lvl.V

            # Smoothed Prolongator must be recomputed because U and V have been updated
            fn, kwargs = unpack_arg(smooth)
            if fn == 'energy':
                Cpt_params = (True, {'I_F': I_F, 'I_C': I_C, 'P_I': P_I, 'Cpts': Cpts})
                B = np.hstack((lvl.U, lvl.V))
                Bc = np.hstack((Uc, Vc))

                # Eventually, we need to scale the candidates, and then do some
                # local SVD compression This will allow use to progressively
                # ignore the candidates in U, and focus on the eigenvectors in V
                B, Bc = scale_candidates(B, Bc, A)

                force_new = True
                if not force_new:  # hasattr(lvl, 'P') and (not compute_aggregation):
                    # It's safe to use previous P as initial guess
                    kwargs2 = dict(kwargs)
                    kwargs2['maxiter'] = 4
                    kwargs2['degree'] = 0
                    P = energy_prolongation_smoother(A, lvl.P, C, Bc, B,
                                                     Cpt_params=Cpt_params, **kwargs2)
                    work[0] += A.nnz*(P.nnz/float(P.shape[0]))*kwargs2['maxiter']
                else:
                    # Start smoothing P from "scratch" with T
                    P = energy_prolongation_smoother(A, T, C, Bc, B,
                                                     Cpt_params=Cpt_params, **kwargs)
                    maxiter = kwargs.get('maxiter', 4)
                    work[0] += A.nnz * (P.nnz / float(P.shape[0])) * maxiter
            else:
                raise ValueError(f'unrecognized prolongation smoother {str(fn)}')

            # Choice of R reflects A's structure
            if symmetry == 'hermitian':
                R = P.H
            elif symmetry == 'symmetric':
                R = P.T
            elif symmetry == 'nonsymmetric':
                R = P.H
                print('Warning nonsymmetric matrix, using R = P.H anyway')

            # Store matrices
            if len(levels) == (l+1):
                levels.append(MultilevelSolver.Level())

            levels[l+1].U = Uc
            levels[l+1].V = Vc
            levels[l+1].A = R*A*P
            levels[l+1].M = R*M*P
            levels[l+1].B = Bc

            lvl.Cnodes = Cnodes
            lvl.Cpts = Cpts
            lvl.I_C = I_C
            lvl.I_F = I_F
            lvl.P_I = P_I
            lvl.B = B
            ##
            lvl.T = T
            lvl.AggOp = AggOp
            lvl.P = P
            lvl.R = R
            lvl.C = C

            # Recursive call -- Updates U and V on all coarser levels
            bootstrap_setup(levels, l+1, aggregate, max_coarse, max_levels, strength,
                            prepostsmoother, smooth, coarse_solver, work, symmetry,
                            mu, candidate_iters, not_frozen, recoarsen, num_eigvects, verbose)

            # Interpolate coarse-grid candidates
            if len(not_frozen) > 0:
                work[0] += A.nnz*candidate_iters*len(not_frozen)*cost(prepostsmoother)
                lvl.U = P*levels[l+1].U
                G = relaxation_as_linear_operator(A, prepostsmoother, candidate_iters)
                lvl.U[:, not_frozen] = G*lvl.U[:, not_frozen]
                for j in not_frozen:
                    alpha = norm(lvl.U[:, j], 'inf')
                    if alpha != 0.0:
                        lvl.U[:, j] /= norm(lvl.U[:, j], 'inf')

            ##
            # Interpolate the coarse grid eigenvector approximations
            lvl.V = P*levels[l+1].V
            if verbose:
                print(f'\nFinishing level {l}, Grid size ({A.shape[0]}, {A.shape[1]})')
            if lvl.V.shape[1] > 0:
                init_eigs = np.zeros((lvl.V.shape[1],), dtype=lvl.V.dtype)
                for j in range(lvl.V.shape[1]):
                    VAV = operator_inner_product(lvl.V[:, j], lvl.V[:, j], A)
                    VMV = operator_inner_product(lvl.V[:, j], lvl.V[:, j], M)
                    init_eigs[j] = VAV / VMV

                # Smooth V and compute the new generalized Rayleigh-Quotient
                work[0] += A.nnz*candidate_iters*lvl.V.shape[1]*cost(prepostsmoother)
                G = eigen_relaxation_as_linear_operator(A, M,
                                                        prepostsmoother, candidate_iters)
                lvl.V = G * lvl.V
                for j in range(lvl.V.shape[1]):
                    alpha = norm(lvl.V[:, j], 'inf')
                    if alpha != 0.0:
                        lvl.V[:, j] /= alpha

                final_eigs = np.zeros((lvl.V.shape[1],), dtype=lvl.V.dtype)
                for j in range(lvl.V.shape[1]):
                    VAV = operator_inner_product(lvl.V[:, j], lvl.V[:, j], A)
                    VMV = operator_inner_product(lvl.V[:, j], lvl.V[:, j], M)
                    final_eigs[j] = VAV / VMV

                # Check for eigenvalue convergence
                conv_test = np.abs(final_eigs - init_eigs) / np.abs(final_eigs)
                rel_change = [f'{ee:1.1e}' for ee in conv_test]
                if verbose:
                    print('Relative change in eig estimate:  ' + str(rel_change))
                    eig_est = [f'{ee.real:1.1e} + {ee.imag:1.1e}' for ee in final_eigs]
                    print('Eigenvalue estimate:              ' + str(eig_est))
