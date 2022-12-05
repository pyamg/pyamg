"""Support for pairwise-aggregation-based AMG."""


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
from .aggregate import pairwise_aggregation
from .tentative import fit_candidates

from ..relaxation.utils import relaxation_as_linear_operator


__all__ = ['pairwise_solver']


def pairwise_solver(A,
                    strength=None,
                    presmoother=('block_gauss_seidel',
                                 {'sweep': 'symmetric'}),
                    postsmoother=('block_gauss_seidel',
                                  {'sweep': 'symmetric'}),
                    max_levels = 20, max_coarse = 10,
                    keep=False, **kwargs):
    """
    Create a multilevel solver using Pairwise Aggregation

    Parameters
    ----------
    A : {csr_matrix, bsr_matrix}
        Sparse NxN matrix in CSR or BSR format
    presmoother : {tuple, string, list} : default ('block_gauss_seidel',
                  {'sweep':'symmetric'})
        Defines the presmoother for the multilevel cycling.  The default block
        Gauss-Seidel option defaults to point-wise Gauss-Seidel, if the matrix
        is CSR or is a BSR matrix with blocksize of 1.  See notes below for
        varying this parameter on a per level basis.
    postsmoother : {tuple, string, list}
        Same as presmoother, except defines the postsmoother.
    max_levels : {integer} : default 10
        Maximum number of levels to be used in the multilevel solver.
    max_coarse : {integer} : default 500
        Maximum number of variables permitted on the coarse grid.
    keep : {bool} : default False
        Flag to indicate keeping extra operators in the hierarchy for
        diagnostics.  For example, if True, then strength of connection (C),
        tentative prolongation (T), and aggregation (AggOp) are kept.

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

    if A.shape[0] != A.shape[1]:
        raise ValueError('expected square matrix')

    # Levelize the user parameters, so that they become lists describing the
    # desired user option on each level.
    max_levels, max_coarse, strength =\
        levelize_strength_or_aggregation(strength, max_levels, max_coarse)

    # Construct multilevel structure
    levels = []
    levels.append(MultilevelSolver.Level())
    levels[-1].A = A          # matrix

    while len(levels) < max_levels and\
            int(levels[-1].A.shape[0]/get_blocksize(levels[-1].A)) > max_coarse:
        _extend_hierarchy(levels, strength, keep)

    ml = MultilevelSolver(levels, **kwargs)
    change_smoothers(ml, presmoother, postsmoother)
    return ml


def _extend_hierarchy(levels, strength, keep):
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

    # Compute pairwise interpolation and restriction matrices, R=P^*
    P = pairwise_aggregation(C, A, compute_P=True, **kwargs)[0]
    R = P.H

    if keep:
        levels[-1].C = C  # strength of connection matrix

    levels[-1].P = P  # smoothed prolongator
    levels[-1].R = R  # restriction operator

    levels.append(MultilevelSolver.Level())
    A = R * A * P              # Galerkin operator
    levels[-1].A = A