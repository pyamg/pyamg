"""Approximate ideal restriction AMG"""
from __future__ import absolute_import

__docformat__ = "restructuredtext en"

from warnings import warn
from scipy.sparse import csr_matrix, bsr_matrix, isspmatrix_csr, isspmatrix_bsr, \
    SparseEfficiencyWarning, block_diag
import numpy as np
from copy import deepcopy

from pyamg.multilevel import multilevel_solver
from pyamg.relaxation.smoothing import change_smoothers
from pyamg.strength import classical_strength_of_connection, \
    symmetric_strength_of_connection, evolution_strength_of_connection, \
    distance_strength_of_connection, algebraic_distance, affinity_distance, \
    energy_based_strength_of_connection
from pyamg.util.utils import unpack_arg, extract_diagonal_blocks, \
    filter_matrix_rows
from pyamg.classical.interpolate import direct_interpolation, \
    standard_interpolation, distance_two_interpolation, injection_interpolation, \
    one_point_interpolation, neumann_AIR, local_AIR
from pyamg.classical.split import RS, PMIS, PMISc, CLJP, CLJPc, MIS
from pyamg.classical.cr import CR

__all__ = ['AIR_solver']

def AIR_solver(A,
               strength=('classical', {'theta': 0.3 ,'norm': 'min'}),
               CF=('RS', {'second_pass': True}),
               interp='one_point',
               restrict=('air', {'theta': 0.05, 'degree': 2}),
               presmoother=None,
               postsmoother=('FC_jacobi', {'omega': 1.0, 'iterations': 1,
                              'withrho': False,  'F_iterations': 2,
                              'C_iterations': 0} ),
               filter_operator=None,
               max_levels=20, max_coarse=20,
               keep=False, **kwargs):
    """Create a multilevel solver using Classical AMG (Ruge-Stuben AMG)

    Parameters
    ----------
    A : csr_matrix
        Square nonsymmetric matrix in CSR format
    strength : ['symmetric', 'classical', 'evolution', 'distance',
                'algebraic_distance','affinity', 'energy_based', None]
        Method used to determine the strength of connection between unknowns
        of the linear system.  Method-specific parameters may be passed in
        using a tuple, e.g. strength=('symmetric',{'theta' : 0.25 }). If
        strength=None, all nonzero entries of the matrix are considered strong.
    CF : {string} : default 'RS'
        Method used for coarse grid selection (C/F splitting)
        Supported methods are RS, PMIS, PMISc, CLJP, CLJPc, and CR.
    interp : {string} : default 'one-point'
        Options include 'direct', 'standard', 'inject' and 'one-point'.
    restrict : {string} : default distance-2 AIR, with theta = 0.05.
        Options include 'air' for local approximate ideal restriction (lAIR)
        and 'neumann' for Neumann approximate ideal restriction (nAIR).
    presmoother : {string or dict} : default None
        Method used for presmoothing at each level.  Method-specific parameters
        may be passed in using a tuple, e.g.
        presmoother=('gauss_seidel',{'sweep':'symmetric}), the default.
    postsmoother : {string or dict} : default F-Jacobi
        Postsmoothing method with the same usage as presmoother
    filter_operator : (bool, tol) : default None
        Remove small entries in operators on each level if True. Entries are
        considered "small" if |a_ij| < tol |a_ii|.
    coarse_grid_P : {string} : default None
        Option to specify a different construction of P used in computing RAP
        vs. for interpolation in an actual solve.
    max_levels: {integer} : default 20
        Maximum number of levels to be used in the multilevel solver.
    max_coarse: {integer} : default 20
        Maximum number of variables permitted on the coarse grid.
    keep: {bool} : default False
        Flag to indicate keeping extra operators in the hierarchy for
        diagnostics.  For example, if True, then strength of connection (C) and
        tentative prolongation (T) are kept.

    Returns
    -------
    ml : multilevel_solver
        Multigrid hierarchy of matrices and prolongation operators

    Other Parameters
    ----------------
    coarse_solver : ['splu', 'lu', 'cholesky, 'pinv', 'gauss_seidel', ... ]
        Solver used at the coarsest level of the MG hierarchy.
            Optionally, may be a tuple (fn, args), where fn is a string such as
        ['splu', 'lu', ...] or a callable function, and args is a dictionary of
        arguments to be passed to fn.

    Notes
    -----




    References
    ----------
    .. [1] 

    See Also
    --------
    aggregation.smoothed_aggregation_solver, multilevel_solver,
    aggregation.rootnode_solver

    """

    # preprocess A
    A = A.asfptype()
    if A.shape[0] != A.shape[1]:
        raise ValueError('expected square matrix')

    levels = [multilevel_solver.level()]
    levels[-1].A = A

    while len(levels) < max_levels and levels[-1].A.shape[0] > max_coarse:
        bottom = extend_hierarchy(levels, strength, CF, interp, restrict, filter_operator,
                                  keep)
        if bottom:
            break

    ml = multilevel_solver(levels, **kwargs)
    change_smoothers(ml, presmoother, postsmoother)
    return ml


# internal function
def extend_hierarchy(levels, strength, CF, interp, restrict, filter_operator, keep):
    """ helper function for local methods """

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
    elif (fn == 'ode') or (fn == 'evolution'):
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
        raise ValueError('unrecognized strength of connection method: %s' %
                         str(fn))

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
    elif fn == 'weighted_matching':
        splitting, soc = weighted_matching(C, **kwargs)
        if soc is not None:
            C = soc
    else:
        raise ValueError('unknown C/F splitting method (%s)' % CF)

    # BS - have run into cases where no C-points are designated, and it
    # throws off the next routines. If everything is an F-point, return here
    if np.sum(splitting) == len(splitting):
        return 1

    # Generate the interpolation matrix that maps from the coarse-grid to the
    # fine-grid
    fn, kwargs = unpack_arg(interp)
    if fn == 'standard':
        P = standard_interpolation(A, C, splitting, **kwargs)
    elif fn == 'distance_two':
        P = distance_two_interpolation(A, C, splitting, **kwargs)
    elif fn == 'direct':
        P = direct_interpolation(A, C, splitting, **kwargs)
    elif fn == 'one_point':
        P = one_point_interpolation(A, C, splitting, **kwargs)
    elif fn == 'inject':
        P = injection_interpolation(A, splitting, **kwargs)
    else:
        raise ValueError('unknown interpolation method (%s)' % interp)

    # Build restriction operator
    fn, kwargs = unpack_arg(restrict)
    if fn is None:
        R = P.T
    elif fn == 'lAIR':
        R = local_AIR(A, splitting, **kwargs)
    elif fn == 'nAIR':
        R = neumann_AIR(A, splitting, **kwargs)
    else:
        raise ValueError('unknown restriction method (%s)' % restrict)

    # Store relevant information for this level
    if keep:
        levels[-1].C = C              # strength of connection matrix

    levels[-1].P = P                  # prolongation operator
    levels[-1].R = R                  # restriction operator
    levels[-1].splitting = splitting  # C/F splitting

    # RAP = R*(A*P)
    A = R * A * P

    # Make sure coarse-grid operator is in correct sparse format
    if (isspmatrix_csr(P) and (not isspmatrix_csr(A))):
        A = A.tocsr()
    elif (isspmatrix_bsr(P) and (not isspmatrix_bsr(A))):
        A = A.tobsr()

    levels.append(multilevel_solver.level())
    levels[-1].A = A
    return 0

