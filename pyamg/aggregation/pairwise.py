"""Support for pairwise-aggregation-based AMG."""


from warnings import warn
import numpy as np
from scipy.sparse import csr_matrix, isspmatrix_csr, isspmatrix_bsr,\
    SparseEfficiencyWarning

from pyamg.multilevel import MultilevelSolver
from pyamg.relaxation.smoothing import change_smoothers
from pyamg.util.utils import get_blocksize, levelize_strength_or_aggregation
from .aggregate import pairwise_aggregation


def pairwise_solver(A,
                    aggregate=('pairwise', {'theta': 0.25,
                               'norm': 'min', 'matchings': 2}),
                    presmoother=('block_gauss_seidel',
                                 {'sweep': 'symmetric'}),
                    postsmoother=('block_gauss_seidel',
                                  {'sweep': 'symmetric'}),
                    max_levels=20, max_coarse=10,
                    **kwargs):
    """
    Create a multilevel solver using Pairwise Aggregation.

    Parameters
    ----------
    A : {csr_matrix, bsr_matrix}
        Sparse NxN matrix in CSR or BSR format
    aggregate : {tuple, string, list} : default ('pairwise',
            {'theta': 0.25, 'norm':'min', 'matchings': 2})
        Method choice must be 'pairwise'; inner pairwise options including
        matchings, theta, and norm can be modified,
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

    Other Parameters
    ----------------
    coarse_solver : ['splu', 'lu', 'cholesky, 'pinv', 'gauss_seidel', ... ]
        Solver used at the coarsest level of the MG hierarchy.
        Optionally, may be a tuple (fn, args), where fn is a string such as
        ['splu', 'lu', ...] or a callable function, and args is a dictionary of
        arguments to be passed to fn.


    Returns
    -------
    ml : multilevel_solver
        Multigrid hierarchy of matrices and prolongation operators

    See Also
    --------
    multilevel_solver, classical.ruge_stuben_solver,
    aggregation.smoothed_aggregation_solver

    Examples
    --------
    >>> from pyamg import pairwise_solver
    >>> from pyamg.gallery import poisson
    >>> from scipy.sparse.linalg import cg
    >>> import numpy as np
    >>> A = poisson((100, 100), format='csr')       # matrix
    >>> b = np.ones((A.shape[0]))                   # RHS
    >>> ml = pairwise_solver(A)                     # AMG solver
    >>> M = ml.aspreconditioner(cycle='V')          # preconditioner
    >>> x, info = cg(A, b, tol=1e-8, maxiter=30, M=M)   # solve with CG

    References
    ----------
    [0] Notay, Y. (2010). An aggregation-based algebraic multigrid
    method. Electronic transactions on numerical analysis, 37(6),
    123-146.

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
    if np.iscomplexobj(A.data):
        raise ValueError('Pairwise solver not verified for complex matrices')

    # Levelize the user parameters, so that they become lists describing the
    # desired user option on each level.
    max_levels, max_coarse, aggregate =\
        levelize_strength_or_aggregation(aggregate, max_levels, max_coarse)

    # Construct multilevel structure
    levels = []
    levels.append(MultilevelSolver.Level())
    levels[-1].A = A          # matrix

    while len(levels) < max_levels and\
            int(levels[-1].A.shape[0]/get_blocksize(levels[-1].A)) > max_coarse:
        _extend_hierarchy(levels, aggregate)

    ml = MultilevelSolver(levels, **kwargs)
    change_smoothers(ml, presmoother, postsmoother)
    return ml


def _extend_hierarchy(levels, aggregate):
    """Extend the multigrid hierarchy."""
    def unpack_arg(v):
        if isinstance(v, tuple):
            return v[0], v[1]
        return v, {}

    A = levels[-1].A

    # Compute pairwise interpolation and restriction matrices, R=P^*
    _, kwargs = unpack_arg(aggregate[len(levels)-1])
    P = pairwise_aggregation(A, **kwargs, compute_P=True)[0]
    R = P.H
    if isspmatrix_csr(P):
        # In this case, R will be CSC, which must be changed
        R = R.tocsr()

    levels[-1].P = P  # unsmoothed prolongator
    levels[-1].R = R  # restriction operator

    levels.append(MultilevelSolver.Level())
    A = R * A * P              # Galerkin operator
    levels[-1].A = A
