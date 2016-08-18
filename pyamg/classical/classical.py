"""Classical AMG (Ruge-Stuben AMG)"""
from __future__ import absolute_import

__docformat__ = "restructuredtext en"

from warnings import warn
from scipy.sparse import csr_matrix, isspmatrix_csr, SparseEfficiencyWarning, block_diag
import numpy

from pyamg.multilevel import multilevel_solver
from pyamg.relaxation.smoothing import change_smoothers
from pyamg.strength import classical_strength_of_connection, \
    symmetric_strength_of_connection, evolution_strength_of_connection,\
    distance_strength_of_connection, energy_based_strength_of_connection,\
    algebraic_distance, affinity_distance
from pyamg.util.utils import extract_diagonal_blocks

from .interpolate import direct_interpolation, \
    standard_interpolation
from . import split

__all__ = ['ruge_stuben_solver']


def ruge_stuben_solver(A,
                       strength=('classical', {'theta': 0.25}),
                       CF='RS',
                       interp='standard',
                       presmoother=('gauss_seidel', {'sweep': 'symmetric'}),
                       postsmoother=('gauss_seidel', {'sweep': 'symmetric'}),
                       max_levels=10, max_coarse=500, keep=False, verts=None, block_starts=None, **kwargs):
    """Create a multilevel solver using Classical AMG (Ruge-Stuben AMG)

    Parameters
    ----------
    A : csr_matrix
        Square matrix in CSR format
    strength : ['symmetric', 'classical', 'evolution', 'algebraic_distance',
                'affinity', None]
        Method used to determine the strength of connection between unknowns
        of the linear system.  Method-specific parameters may be passed in
        using a tuple, e.g. strength=('symmetric',{'theta' : 0.25 }). If
        strength=None, all nonzero entries of the matrix are considered strong.
    CF : {string} : default 'RS'
        Method used for coarse grid selection (C/F splitting)
        Supported methods are RS, PMIS, PMISc, CLJP, and CLJPc
    presmoother : {string or dict}
        Method used for presmoothing at each level.  Method-specific parameters
        may be passed in using a tuple, e.g.
        presmoother=('gauss_seidel',{'sweep':'symmetric}), the default.
    postsmoother : {string or dict}
        Postsmoothing method with the same usage as presmoother
    max_levels: {integer} : default 10
        Maximum number of levels to be used in the multilevel solver.
    max_coarse: {integer} : default 500
        Maximum number of variables permitted on the coarse grid.
    keep: {bool} : default False
        Flag to indicate keeping extra operators in the hierarchy for
        diagnostics.  For example, if True, then strength of connection (C) and
        tentative prolongation (T) are kept.
    verts: array of tuples
        Physical locations of dofs. Used for visualizing coarse grids.
    block_starts: list of integers
        If non-trivial, list of starting row indices of blocks of A if A represents a system (used for unknown approach for systems)

    Returns
    -------
    ml : multilevel_solver
        Multigrid hierarchy of matrices and prolongation operators

    Examples
    --------
    >>> from pyamg.gallery import poisson
    >>> from pyamg import ruge_stuben_solver
    >>> A = poisson((10,),format='csr')
    >>> ml = ruge_stuben_solver(A,max_coarse=3)

    Notes
    -----

    "coarse_solver" is an optional argument and is the solver used at the
    coarsest grid.  The default is a pseudo-inverse.  Most simply,
    coarse_solver can be one of ['splu', 'lu', 'cholesky, 'pinv',
    'gauss_seidel', ... ].  Additionally, coarse_solver may be a tuple
    (fn, args), where fn is a string such as ['splu', 'lu', ...] or a callable
    function, and args is a dictionary of arguments to be passed to fn.


    References
    ----------
    .. [1] Trottenberg, U., Oosterlee, C. W., and Schuller, A.,
       "Multigrid" San Diego: Academic Press, 2001.  Appendix A

    See Also
    --------
    aggregation.smoothed_aggregation_solver, multilevel_solver,
    aggregation.rootnode_solver

    """

    levels = [multilevel_solver.level()]

    # convert A to csr
    if not ( isspmatrix_csr(A) ):
        try:
            A = csr_matrix(A)
            warn("Implicit conversion of A to CSR",
                 SparseEfficiencyWarning)
        except:
            raise TypeError('Argument A must have type csr_matrix, \
                             or be convertible to csr_matrix')
    # preprocess A
    A = A.asfptype()
    if A.shape[0] != A.shape[1]:
        raise ValueError('expected square matrix')

    levels[-1].A = A
    levels[-1].block_starts = block_starts
    if verts is None:
        levels[-1].verts = numpy.zeros(1)
    else:
        levels[-1].verts = verts

    while len(levels) < max_levels and levels[-1].A.shape[0] > max_coarse:
        # print "Level ", len(levels) - 1
        extend_hierarchy(levels, strength, CF, interp, keep)

    ml = multilevel_solver(levels, **kwargs)
    change_smoothers(ml, presmoother, postsmoother)
    return ml


# internal function
def extend_hierarchy(levels, strength, CF, interp, keep):
    """ helper function for local methods """
    def unpack_arg(v):
        if isinstance(v, tuple):
            return v[0], v[1]
        else:
            return v, {}

    A = levels[-1].A
    block_starts = levels[-1].block_starts
    verts = levels[-1].verts

    # If this is a system, apply the unknown approach by coarsening and generating interpolation based on each diagonal block of A
    if (block_starts):
        A_diag = extract_diagonal_blocks(A, block_starts)
    else:
        A_diag = [A]

    # Compute the strength-of-connection matrix C, where larger
    # C[i,j] denote stronger couplings between i and j.
    C_diag = []
    P_diag = []
    splitting = []
    next_lvl_block_starts = [0]
    block_cnt = 0
    for mat in A_diag:
        fn, kwargs = unpack_arg(strength)
        if fn == 'symmetric':
            C_diag.append( symmetric_strength_of_connection(mat, **kwargs) )
        elif fn == 'classical':
            C_diag.append( classical_strength_of_connection(mat, **kwargs) )
        elif fn == 'distance':
            C_diag.append( distance_strength_of_connection(mat, **kwargs) )
        elif (fn == 'ode') or (fn == 'evolution'):
            C_diag.append( evolution_strength_of_connection(mat, **kwargs) )
        elif fn == 'energy_based':
            C_diag.append( energy_based_strength_of_connection(mat, **kwargs) )
        elif fn == 'algebraic_distance':
            C_diag.append( algebraic_distance(mat, **kwargs) )
        elif fn == 'affinity':
            C_diag.append( affinity_distance(mat, **kwargs) )
        elif fn is None:
            C_diag.append( mat )
        else:
            raise ValueError('unrecognized strength of connection method: %s' %
                             str(fn))

        # Generate the C/F splitting
        fn, kwargs = unpack_arg(CF)
        if fn == 'RS':
            splitting.append( split.RS(C_diag[-1]) )
        elif fn == 'PMIS':
            splitting.append( split.PMIS(C_diag[-1]) )
        elif fn == 'PMISc':
            splitting.append( split.PMISc(C_diag[-1]) )
        elif fn == 'CLJP':
            splitting.append( split.CLJP(C_diag[-1]) )
        elif fn == 'CLJPc':
            splitting.append( split.CLJPc(C_diag[-1]) )
        elif fn == 'Shifted2DCoarsening':
            splitting.append( split.Shifted2DCoarsening(C_diag[-1]) )
        else:
            raise ValueError('unknown C/F splitting method (%s)' % CF)

        # Generate the interpolation matrix that maps from the coarse-grid to the
        # fine-grid
        fn, kwargs = unpack_arg(interp)
        if fn == 'standard':
            P_diag.append( standard_interpolation(mat, C_diag[-1], splitting[-1]) )
        elif fn == 'direct':
            P_diag.append( direct_interpolation(mat, C_diag[-1], splitting[-1]) )
        else:
            raise ValueError('unknown interpolation method (%s)' % interp)

        next_lvl_block_starts.append( next_lvl_block_starts[-1] + P_diag[-1].shape[1])

        block_cnt = block_cnt + 1

    P = block_diag(P_diag)

    # Generate the restriction matrix that maps from the fine-grid to the
    # coarse-grid
    R = P.T.tocsr()

    # Store relevant information for this level
    splitting = numpy.concatenate(splitting)
    if keep:
        C = block_diag(C_diag)
        levels[-1].C = C                  # strength of connection matrix
        levels[-1].splitting = splitting  # C/F splitting

    levels[-1].P = P                  # prolongation operator
    levels[-1].R = R                  # restriction operator

    levels.append(multilevel_solver.level())

    # Form next level through Galerkin product
    # !!! For systems, how do I propogate the block structure information down to the next grid? Especially if the blocks are different sizes? !!!
    A = R * A * P
    levels[-1].A = A

    if (block_starts):
        levels[-1].block_starts = next_lvl_block_starts
    else:
        levels[-1].block_starts = None

    # If called for, output a visualization of the C/F splitting
    if (verts.any()):
        new_verts = numpy.empty([P.shape[1], 2])
        cnt = 0
        for i in range(len(splitting)):
            if (splitting[i]):
                new_verts[cnt] = verts[i]
                cnt = cnt + 1
        levels[-1].verts = new_verts
    else:
        levels[-1].verts = numpy.zeros(1)
