"""Approximate idealkrestriction AMG."""

from copy import deepcopy
import numpy as np
from scipy.sparse import isspmatrix_csr, isspmatrix_bsr

from ..multilevel import MultilevelSolver
from ..relaxation.smoothing import change_smoothers
from ..strength import (classical_strength_of_connection,
                        symmetric_strength_of_connection, evolution_strength_of_connection,
                        distance_strength_of_connection, algebraic_distance,
                        affinity_distance, energy_based_strength_of_connection)
from ..util.utils import filter_matrix_rows
from ..classical.interpolate import (direct_interpolation, classical_interpolation,
                                     injection_interpolation, one_point_interpolation,
                                     local_air)
from .split import RS, PMIS, PMISc, CLJP, CLJPc
from .cr import CR


def air_solver(A,
               strength=('classical', {'theta': 0.3, 'norm': 'min'}),
               CF=('RS', {'second_pass': True}),
               interpolation='one_point',
               restrict=('air', {'theta': 0.05, 'degree': 2}),
               presmoother=None,
               postsmoother=('fc_jacobi', {'omega': 1.0, 'iterations': 1,
                                           'withrho': False, 'f_iterations': 2,
                                           'c_iterations': 1}),
               filter_operator=None,
               max_levels=20, max_coarse=20,
               keep=False, **kwargs):
    """Create a multilevel solver using approximate ideal restriction (AIR) AMG.

    Parameters
    ----------
    A : csr_matrix
        Square (non)symmetric matrix in CSR format
    strength : ['symmetric', 'classical', 'evolution', 'distance',
                'algebraic_distance','affinity', 'energy_based', None]
        Method used to determine the strength of connection between unknowns
        of the linear system.  Method-specific parameters may be passed in
        using a tuple, e.g. strength=('symmetric',{'theta' : 0.25 }). If
        strength=None, all nonzero entries of the matrix are considered strong.
    CF : {string} : default 'RS' with second pass
        Method used for coarse grid selection (C/F splitting)
        Supported methods are RS, PMIS, PMISc, CLJP, CLJPc, and CR.
    interpolation : {string} : default 'one_point'
        Options include 'direct', 'classical', 'inject' and 'one-point'.
    restrict : {string} : default distance-2 AIR, with theta = 0.05.
        Option is 'air' for local approximate ideal restriction (lAIR),
        with inner options specifying degree, strength tolerance, etc.
    presmoother : {string or dict} : default None
        Method used for presmoothing at each level.  Method-specific parameters
        may be passed in using a tuple.
    postsmoother : {string or dict}
        Postsmoothing method with the same usage as presmoother.
        postsmoother=('fc_jacobi', ... ) with 2 F-sweeps, 1 C-sweep is default.
    filter_operator : (bool, tol) : default None
        Remove small entries in operators on each level if True. Entries are
        considered "small" if |a_ij| < tol |a_ii|.
    max_levels: {integer} : default 20
        Maximum number of levels to be used in the multilevel solver.
    max_coarse: {integer} : default 20
        Maximum number of variables permitted on the coarse grid.
    keep: {bool} : default False
        Flag to indicate keeping strength of connection matrix (C) in
        hierarchy.

    Returns
    -------
    ml : multilevel_solver
        Multigrid hierarchy of matrices and prolongation operators

    Examples
    --------
    >>> from pyamg.gallery import poisson
    >>> from pyamg import air_solver
    >>> A = poisson((10,),format='csr')
    >>> ml = air_solver(A,max_coarse=3)

    Notes
    -----
    "coarse_solver" is an optional argument and is the solver used at the
    coarsest grid.  The default is a pseudo-inverse.  Most simply,
    coarse_solver can be one of ['splu', 'lu', 'cholesky, 'pinv',
    'gauss_seidel', ... ].  Additionally, coarse_solver may be a tuple
    (fn, args), where fn is a string such as ['splu', 'lu', ...] or a callable
    function, and args is a dictionary of arguments to be passed to fn.
    See [2001TrOoSc]_ for additional details.


    References
    ----------
    [1] Manteuffel, T. A., Münzenmaier, S., Ruge, J., & Southworth, B. S.
    (2019). Nonsymmetric reduction-based algebraic multigrid. SIAM
    Journal on Scientific Computing, 41(5), S242-S268.

    [2] Manteuffel, T. A., Ruge, J., & Southworth, B. S. (2018).
    Nonsymmetric algebraic multigrid based on local approximate ideal
    restriction (lAIR). SIAM Journal on Scientific Computing, 40(6),
    A4105-A4130.

    See Also
    --------
    aggregation.smoothed_aggregation_solver, multilevel_solver,
    aggregation.rootnode_solver, ruge_stuben_solver

    """
    # preprocess A
    A = A.asfptype()
    if A.shape[0] != A.shape[1]:
        raise ValueError('expected square matrix')
    if np.iscomplexobj(A.data):
        raise ValueError('AIR solver not verified for complex matrices')

    levels = [MultilevelSolver.Level()]
    levels[-1].A = A

    while len(levels) < max_levels and levels[-1].A.shape[0] > max_coarse:
        bottom = extend_hierarchy(levels, strength, CF, interpolation, restrict,
                                  filter_operator, keep)
        if bottom:
            break

    ml = MultilevelSolver(levels, **kwargs)
    change_smoothers(ml, presmoother, postsmoother)
    return ml


def extend_hierarchy(levels, strength, CF, interpolation, restrict, filter_operator, keep):
    """Extend the multigrid hierarchy."""
    def unpack_arg(v):
        if isinstance(v, tuple):
            return v[0], v[1]
        return v, {}

    # Filter operator. Need to keep original matrix on finest level for
    # computing residuals
    if (filter_operator is not None) and (filter_operator[1] != 0):
        if len(levels) == 1:
            A = deepcopy(levels[-1].A)
        else:
            A = levels[-1].A
        filter_matrix_rows(A, filter_operator[1], diagonal=True, lump=filter_operator[0])
    else:
        A = levels[-1].A

    # Check if matrix was filtered to be diagonal --> coarsest grid
    if A.nnz == A.shape[0]:
        return 1

    # Compute the strength-of-connection matrix C, where larger
    # C[i,j] denote stronger couplings between i and j.
    fn, kwargs = unpack_arg(strength)
    if fn == 'symmetric':
        C = symmetric_strength_of_connection(A, **kwargs)
    elif fn == 'classical':
        C = classical_strength_of_connection(A, **kwargs)
    elif fn == 'distance':
        C = distance_strength_of_connection(A, **kwargs)
    elif fn in ('ode', 'evolution'):
        C = evolution_strength_of_connection(A, **kwargs)
    elif fn == 'energy_based':
        C = energy_based_strength_of_connection(A, **kwargs)
    elif fn == 'algebraic_distance':
        C = algebraic_distance(A, **kwargs)
    elif fn == 'affinity':
        C = affinity_distance(A, **kwargs)
    elif fn is None:
        C = A
    else:
        raise ValueError(f'Unrecognized strength of connection method: {fn}')

    # Generate the C/F splitting
    fn, kwargs = unpack_arg(CF)
    if fn == 'RS':
        splitting = RS(C, **kwargs)
    elif fn == 'PMIS':
        splitting = PMIS(C, **kwargs)
    elif fn == 'PMISc':
        splitting = PMISc(C, **kwargs)
    elif fn == 'CLJP':
        splitting = CLJP(C, **kwargs)
    elif fn == 'CLJPc':
        splitting = CLJPc(C, **kwargs)
    elif fn == 'CR':
        splitting = CR(C, **kwargs)
    else:
        raise ValueError(f'Unknown C/F splitting method {CF}')

    # Make sure all points were not declared as C- or F-points
    num_fpts = np.sum(splitting)
    if (num_fpts == len(splitting)) or (num_fpts == 0):
        return 1

    # Generate the interpolation matrix that maps from the coarse-grid to the
    # fine-grid
    fn, kwargs = unpack_arg(interpolation)
    if fn == 'classical':
        P = classical_interpolation(A, C, splitting, **kwargs)
    elif fn == 'direct':
        P = direct_interpolation(A, C, splitting, **kwargs)
    elif fn == 'one_point':
        P = one_point_interpolation(A, C, splitting, **kwargs)
    elif fn == 'inject':
        P = injection_interpolation(A, splitting, **kwargs)
    else:
        raise ValueError(f'Unknown interpolation method {fn}')

    # Build restriction operator
    fn, kwargs = unpack_arg(restrict)
    if fn == 'air':
        R = local_air(A, splitting, **kwargs)
    else:
        raise ValueError(f'Unknown restriction method {fn}')

    # Store relevant information for this level
    if keep:
        levels[-1].C = C              # strength of connection matrix

    levels[-1].splitting = splitting.astype(bool)  # C/F splitting
    levels[-1].P = P                               # prolongation operator
    levels[-1].R = R                               # restriction operator

    # RAP = R*(A*P)
    A = R * A * P

    # Make sure coarse-grid operator is in correct sparse format
    if (isspmatrix_csr(P) and (not isspmatrix_csr(A))):
        A = A.tocsr()
    elif (isspmatrix_bsr(P) and (not isspmatrix_bsr(A))):
        A = A.tobsr()

    levels.append(MultilevelSolver.Level())
    levels[-1].A = A
    return 0
