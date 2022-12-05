"""Support for aggregation-based AMG."""


from warnings import warn
import numpy as np
from scipy.sparse import csr_matrix, isspmatrix_csr, isspmatrix_bsr,\
    SparseEfficiencyWarning

from pyamg.multilevel import MultilevelSolver
from pyamg.relaxation.smoothing import change_smoothers
from pyamg.util.utils import eliminate_diag_dom_nodes, get_blocksize,\
    levelize_strength_or_aggregation, levelize_smooth_or_improve_candidates
from pyamg.strength import classical_strength_of_connection,\
    symmetric_strength_of_connection, evolution_strength_of_connection,\
    energy_based_strength_of_connection, distance_strength_of_connection,\
    algebraic_distance, affinity_distance
from .aggregate import standard_aggregation, naive_aggregation,\
    lloyd_aggregation, notay_pairwise, weighted_matching
from .tentative import fit_candidates
from .smooth import jacobi_prolongation_smoother,\
    richardson_prolongation_smoother, energy_prolongation_smoother

from ..relaxation.utils import relaxation_as_linear_operator


__all__ = ['pairwise_solver']


def pairwise_solver(A, B=None, BH=None,
                    symmetry='hermitian',
                    strength=None,
                    aggregate='standard',
                    presmoother=('block_gauss_seidel',
                                 {'sweep': 'symmetric'}),
                    postsmoother=('block_gauss_seidel',
                                  {'sweep': 'symmetric'}),
                    improve_candidates=[('block_gauss_seidel',
                                        {'sweep': 'symmetric',
                                         'iterations': 4}), None],
                    max_levels = 20, max_coarse = 10,
                    keep=False, **kwargs):
    """
    Create a multilevel solver using Pairwise Aggregation

    Parameters
    ----------
    A : {csr_matrix, bsr_matrix}
        Sparse NxN matrix in CSR or BSR format
    B : {None, array_like}
        Right near-nullspace candidates stored in the columns of an NxK array.
        The default value B=None is equivalent to B=ones((N,1))
    BH : {None, array_like}
        Left near-nullspace candidates stored in the columns of an NxK array.
        BH is only used if symmetry is 'nonsymmetric'.
        The default value B=None is equivalent to BH=B.copy()
    symmetry : {string}
        'symmetric' refers to both real and complex symmetric
        'hermitian' refers to both complex Hermitian and real Hermitian
        'nonsymmetric' i.e. nonsymmetric in a hermitian sense
        Note, in the strictly real case, symmetric and hermitian are the same
        Note, this flag does not denote definiteness of the operator.
    aggregate : {list} : default ['standard', 'lloyd', 'naive', 'pairwise',
                ('predefined', {'AggOp' : csr_matrix})]
        Method used to aggregate nodes.  See notes below for varying this
        parameter on a per level basis.  Also, see notes below for using a
        predefined aggregation on each level. Method-specific parameters may be
        passed in using a tuple, e.g. aggregate=('pairwise',{'num_matchings': 2 })
    presmoother : {tuple, string, list} : default ('block_gauss_seidel',
                  {'sweep':'symmetric'})
        Defines the presmoother for the multilevel cycling.  The default block
        Gauss-Seidel option defaults to point-wise Gauss-Seidel, if the matrix
        is CSR or is a BSR matrix with blocksize of 1.  See notes below for
        varying this parameter on a per level basis.
    postsmoother : {tuple, string, list}
        Same as presmoother, except defines the postsmoother.
    improve_candidates : {tuple, string, list} : default
                        [('block_gauss_seidel',
                         {'sweep': 'symmetric', 'iterations': 4}), None]
        The ith entry defines the method used to improve the candidates B on
        level i.  If the list is shorter than max_levels, then the last entry
        will define the method for all levels lower.  If tuple or string, then
        this single relaxation descriptor defines improve_candidates on all
        levels.
        The list elements are relaxation descriptors of the form used for
        presmoother and postsmoother.  A value of None implies no action on B.
    max_levels : {integer} : default 10
        Maximum number of levels to be used in the multilevel solver.
    max_coarse : {integer} : default 500
        Maximum number of variables permitted on the coarse grid.
    keep : {bool} : default False
        Flag to indicate keeping extra operators in the hierarchy for
        diagnostics.  For example, if True, then strength of connection (C),
        tentative prolongation (T), and aggregation (AggOp) are kept.

    Other Parameters
    ----------------
    cycle_type : ['V','W','F']
        Structrure of multigrid cycle
    coarse_solver : ['splu', 'lu', 'cholesky, 'pinv', 'gauss_seidel', ... ]
        Solver used at the coarsest level of the MG hierarchy.
            Optionally, may be a tuple (fn, args), where fn is a string such as
        ['splu', 'lu', ...] or a callable function, and args is a dictionary of
        arguments to be passed to fn.
    setup_complexity : bool
        For a detailed, more accurate setup complexity, pass in 
        'setup_complexity' = True. This will slow down performance, but
        increase accuracy of complexity count. 

    Returns
    -------
    ml : multilevel_solver
        Multigrid hierarchy of matrices and prolongation operators

    See Also
    --------
    multilevel_solver, classical.ruge_stuben_solver,
    aggregation.smoothed_aggregation_solver

    Notes
    -----



    References
    ----------



    """
    if not (isspmatrix_csr(A) or isspmatrix_bsr(A)):
        try:
            A = csr_matrix(A)
            warn('Implicit conversion of A to CSR', SparseEfficiencyWarning)
        except BaseException as e:
            raise TypeError('Argument A must have type csr_matrix or bsr_matrix, '
                            'or be convertible to csr_matrix') from e

    A = A.asfptype()

    if symmetry not in ('symmetric', 'hermitian', 'nonsymmetric'):
        raise ValueError('Expected "symmetric", "nonsymmetric" or "hermitian" '
                         'for the symmetry parameter ')
    A.symmetry = symmetry

    if A.shape[0] != A.shape[1]:
        raise ValueError('expected square matrix')

    # Right near nullspace candidates use constant for each variable as default
    if B is None:
        B = np.kron(np.ones((int(A.shape[0]/get_blocksize(A)), 1), dtype=A.dtype),
                    np.eye(get_blocksize(A), dtype=A.dtype))
    else:
        B = np.asarray(B, dtype=A.dtype)
        if len(B.shape) == 1:
            B = B.reshape(-1, 1)
        if B.shape[0] != A.shape[0]:
            raise ValueError('The shape of near null-space modes B is incorrect')
        if B.shape[1] < get_blocksize(A):
            warn('Having less target vectors, B.shape[1], than blocksize of A '
                 'can degrade convergence factors.')

    # Left near nullspace candidates
    if A.symmetry == 'nonsymmetric':
        if BH is None:
            BH = B.copy()
        else:
            BH = np.asarray(BH, dtype=A.dtype)
            if len(BH.shape) == 1:
                BH = BH.reshape(-1, 1)
            if BH.shape[1] != B.shape[1]:
                raise ValueError('The number of left and right near null-space '
                                 'modes B and BH must be equal')
            if BH.shape[0] != A.shape[0]:
                raise ValueError('The shape of near null-space modes B is incorrect')

    # Levelize the user parameters, so that they become lists describing the
    # desired user option on each level.
    max_levels, max_coarse, strength =\
        levelize_strength_or_aggregation(strength, max_levels, max_coarse)
    max_levels, max_coarse, aggregate =\
        levelize_strength_or_aggregation(aggregate, max_levels, max_coarse)
    improve_candidates =\
        levelize_smooth_or_improve_candidates(improve_candidates, max_levels)
    smooth = levelize_smooth_or_improve_candidates(smooth, max_levels)

    # Construct multilevel structure
    levels = []
    levels.append(MultilevelSolver.Level())
    levels[-1].A = A          # matrix

    # Append near nullspace candidates
    levels[-1].B = B          # right candidates
    if A.symmetry == 'nonsymmetric':
        levels[-1].BH = BH    # left candidates

    while len(levels) < max_levels and\
            int(levels[-1].A.shape[0]/get_blocksize(levels[-1].A)) > max_coarse:
        _extend_hierarchy(levels, strength, aggregate,
                          improve_candidates, keep)

    ml = MultilevelSolver(levels, **kwargs)
    change_smoothers(ml, presmoother, postsmoother)
    return ml


def _extend_hierarchy(levels, strength, aggregate, improve_candidates, keep):
    """Extend the multigrid hierarchy.

    Service routine to implement the strength of connection, aggregation,
    tentative prolongation construction, and prolongation smoothing.  Called by
    smoothed_aggregation_solver.

    """
    def unpack_arg(v):
        if isinstance(v, tuple):
            return v[0], v[1]
        return v, {}

    A = levels[-1].A
    B = levels[-1].B
    if A.symmetry == "nonsymmetric":
        AH = A.H.asformat(A.format)
        BH = levels[-1].BH

    # Compute the strength-of-connection matrix C, where larger
    # C[i,j] denote stronger couplings between i and j. Note, this
    # is not used for aggregation, only as sparsity pattern for
    # energy-min smoothing, or filtering Jacobi smoothing of T.
    fn, kwargs = unpack_arg(strength[len(levels)-1])
    if fn == 'symmetric':
        C = symmetric_strength_of_connection(A, **kwargs)
    elif fn == 'classical':
        C = classical_strength_of_connection(A, **kwargs)
    elif fn == 'distance':
        C = distance_strength_of_connection(A, **kwargs)
    elif (fn == 'ode') or (fn == 'evolution'):
        if 'B' in kwargs:
            C = evolution_strength_of_connection(A, **kwargs)
        else:
            C = evolution_strength_of_connection(A, B, **kwargs)
    elif fn == 'energy_based':
        C = energy_based_strength_of_connection(A, **kwargs)
    elif fn == 'predefined':
        C = kwargs['C'].tocsr()
    elif fn == 'algebraic_distance':
        C = algebraic_distance(A, **kwargs)
    elif fn == 'affinity':
        C = affinity_distance(A, **kwargs)
    elif fn is None:
        C = A.tocsr()
    else:
        raise ValueError('unrecognized strength of connection method: %s' %
                         str(fn))

    # Get choice of aggregation routine and arguments
    fn, kwargs = unpack_arg(aggregate[len(levels)-1])

    # Check for boolean "improve_candidates." If true, target vectors are
    # improved each level of pairwise aggregation with chosen relaxation scheme. 
    if ('improve_candidates' in kwargs) and \
       (kwargs['improve_candidates'] == True):
       kwargs['improve_candidates'] = improve_candidates[len(levels)-1]
    else:
       kwargs['improve_candidates'] = None

    # Improve near nullspace candidates by relaxing on A B = 0
    fn, kwargs = unpack_arg(improve_candidates[len(levels)-1])
    if fn is not None:
        b = np.zeros((A.shape[0], 1), dtype=A.dtype)
        B = relaxation_as_linear_operator((fn, kwargs), A, b) * B
        levels[-1].B = B
        if A.symmetry == 'nonsymmetric':
            BH = relaxation_as_linear_operator((fn, kwargs), AH, b) * BH
            levels[-1].BH = BH


    # Compute tentative interpolation operator T. Only fits one target
    # per aggregate - if more are provided, they are fit in fit_candidates().
    if fn == 'notay':
        P = notay_pairwise(A, B=B, **kwargs)
    elif fn == 'matching':
        P = weighted_matching(A, B=B, **kwargs)
    else:
        raise ValueError('unrecognized aggregation method %s' % str(fn))

    # TODO : Need option for B=None because this saves cost in computing
    #        pairwise... 


    ### GENERAL TODO :
    ### - Generalize current Notay PR to support block matrix, candidate 
    ### vectors, and multiple rounds of aggregation/matching.
    



    # Compute the restriction matrix, R, which interpolates from the fine-grid
    # to the coarse-grid.  If A is nonsymmetric, then R must be constructed
    # based on A.H.  Otherwise R = P.H or P.T.
    symmetry = A.symmetry
    if symmetry == 'hermitian':
        R = P.H
    elif symmetry == 'symmetric':
        R = P.T
    elif symmetry == 'nonsymmetric':
        R = T.H

    if keep:
        levels[-1].C = C  # strength of connection matrix
        levels[-1].AggOp = AggOp  # aggregation operator
        levels[-1].T = T  # tentative prolongator

    levels[-1].P = P  # smoothed prolongator
    levels[-1].R = R  # restriction operator

    levels.append(MultilevelSolver.Level())
    A = R * A * P              # Galerkin operator
    A.symmetry = symmetry
    levels[-1].A = A
    levels[-1].B = B           # right near nullspace candidates

    if A.symmetry == 'nonsymmetric':
        levels[-1].BH = BH     # left near nullspace candidates
