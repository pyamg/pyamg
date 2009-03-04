"""Support for aggregation-based AMG"""

__docformat__ = "restructuredtext en"

import numpy
from scipy.sparse import csr_matrix, isspmatrix_csr, isspmatrix_bsr

from pyamg import amg_core
from pyamg.multilevel import multilevel_solver
from pyamg.relaxation.smoothing import setup_smoothers
from pyamg.util.utils import symmetric_rescaling, diag_sparse
from pyamg.strength import classical_strength_of_connection, \
        symmetric_strength_of_connection, ode_strength_of_connection, \
        energy_based_strength_of_connection

from aggregate import standard_aggregation, lloyd_aggregation
from tentative import fit_candidates
from smooth import jacobi_prolongation_smoother, richardson_prolongation_smoother, \
        energy_prolongation_smoother, kaczmarz_richardson_prolongation_smoother, \
        kaczmarz_jacobi_prolongation_smoother

__all__ = ['smoothed_aggregation_solver']

def smoothed_aggregation_solver(A, B=None, 
        mat_flag='hermitian',
        strength='symmetric', 
        aggregate='standard', 
        smooth=('jacobi', {'omega': 4.0/3.0}),
        presmoother=('gauss_seidel',{'sweep':'symmetric'}),
        postsmoother=('gauss_seidel',{'sweep':'symmetric'}),
        max_levels = 10, max_coarse = 500, **kwargs):
    """
    Create a multilevel solver using Smoothed Aggregation (SA)

    Parameters
    ----------
    A : {csr_matrix, bsr_matrix}
        Sparse NxN matrix in CSR or BSR format
    B : {None, array_like}
        Near-nullspace candidates stored in the columns of an NxK array.
        The default value B=None is equivalent to B=ones((N,1))
    mat_flag : {string}
        'symmetric' refers to both real and complex symmetric
        'hermitian' refers to both complex Hermitian and real Hermitian
        Note that for the strictly real case, these two options are the same
        Note that this flag does not denote definiteness of the operator
    strength : ['symmetric', 'classical', 'ode', None]
        Method used to determine the strength of connection between unknowns
        of the linear system.  Method-specific parameters may be passed in
        using a tuple, e.g. strength=('symmetric',{'theta' : 0.25 }). If
        strength=None, all nonzero entries of the matrix are considered strong.
    aggregate : ['standard', 'lloyd', ('predefined',[csr_matrix, ...])]
        Method used to aggregate nodes.  A predefined aggregation is specified
        with a sequence of csr_matrices that represent the aggregation
        operators on each level of the hierarchy.  For instance [ Agg0, Agg1 ]
        defines a three-level hierarchy where the dimensions of A, Agg0 and
        Agg1 are compatible, i.e.  Agg0.shape[1] == A.shape[0] and
        Agg1.shape[1] == Agg0.shape[0].
    smooth : ['jacobi', 'richardson', 'energy', 'kaczmarz_jacobi', 'kaczmarz_richardson', None]
        Method used used to smooth the tentative prolongator.
    max_levels : {integer} : default 10
        Maximum number of levels to be used in the multilevel solver.
    max_coarse : {integer} : default 500
        Maximum number of variables permitted on the coarse grid.

    Other Parameters
    ----------------
    cycle_type : ['V','W','F']
        Structrure of multigrid cycle
    presmoother  : ['gauss_seidel', 'jacobi', ... ]
        Premoother used during multigrid cycling
    postsmoother : ['gauss_seidel', 'jacobi', ... ]
        Postmoother used during multigrid cycling
    coarse_solver : ['splu','lu', ... ]
        Solver used at the coarsest level of the MG hierarchy 

    Returns
    -------
    ml : multilevel_solver
        Multigrid hierarchy of matrices and prolongation operators

    See Also
    --------
    multilevel_solver, classical.ruge_stuben_solver

    Notes
    -----
    - The additional parameters are passed through as arguments to
      multilevel_solver.  Refer to pyamg.multilevel_solver for additional
      documentation.

    - At each level, four steps are executed in order to define the coarser
      level operator.

      1. Matrix A is given and used to derive a strength matrix, C.
      2. Based on the strength matrix, indices are grouped or aggregated.
      3. The aggregates define coarse nodes and a tentative prolongation 
         operator T is defined by injection 
      4. The tentative prolongation operator is smoothed by a relaxation
         scheme to improve the quality and extent of interpolation from the
         aggregates to fine nodes.

    Examples
    --------
    >>> from pyamg import smoothed_aggregation_solver
    >>> from pyamg.gallery import poisson
    >>> from scipy.sparse.linalg import cg
    >>> import numpy
    >>> A = poisson((100,100), format='csr')           # matrix
    >>> b = numpy.ones((A.shape[0]))                         # random RHS
    >>> ml = smoothed_aggregation_solver(A)            # AMG solver
    >>> M = ml.aspreconditioner(cycle='V')             # preconditioner
    >>> x,info = cg(A, b, tol=1e-8, maxiter=30, M=M)   # solve with CG

    References
    ----------
    .. [1] Vanek, P. and Mandel, J. and Brezina, M., 
       "Algebraic Multigrid by Smoothed Aggregation for 
       Second and Fourth Order Elliptic Problems", 
       Computing, vol. 56, no. 3, pp. 179--196, 1996.
       http://citeseer.ist.psu.edu/vanek96algebraic.html

    """
    if not (isspmatrix_csr(A) or isspmatrix_bsr(A)):
        raise TypeError('argument A must have type csr_matrix or bsr_matrix')

    A = A.asfptype()
    
    if (mat_flag != 'symmetric') and (mat_flag != 'hermitian'):
        raise ValueError('expected symmetric or hermitian mat_flag')
    A.symmetry = mat_flag

    if A.shape[0] != A.shape[1]:
        raise ValueError('expected square matrix')

    if B is None:
        B = numpy.ones((A.shape[0],1), dtype=A.dtype) # use constant vector
    else:
        B = numpy.asarray(B, dtype=A.dtype)
    
    if isinstance(aggregate,tuple) and aggregate[0] == 'predefined':
        # predefined aggregation operators
        max_levels = len(aggregate[1]) + 1
        max_coarse = 0

    levels = []
    levels.append( multilevel_solver.level() )
    levels[-1].A = A          # matrix
    levels[-1].B = B          # near-nullspace candidates

    while len(levels) < max_levels and levels[-1].A.shape[0] > max_coarse:
        extend_hierarchy(levels, strength, aggregate, smooth)
    
    ml = multilevel_solver(levels, **kwargs)
    setup_smoothers(ml, presmoother, postsmoother)
    return ml

def extend_hierarchy(levels, strength, aggregate, smooth):
    """Service routine to implement the strenth of connection, aggregation,
    tentative prolongation construction, and prolongation smoothing.  Called by
    smoothed_aggregation_solver.
    """

    def unpack_arg(v):
        if isinstance(v,tuple):
            return v[0],v[1]
        else:
            return v,{}

    A = levels[-1].A
    B = levels[-1].B

    ##
    # strength of connection
    fn, kwargs = unpack_arg(strength)
    if fn == 'symmetric':
        C = symmetric_strength_of_connection(A, **kwargs)
    elif fn == 'classical':
        C = classical_strength_of_connection(A, **kwargs)
    elif fn == 'ode':
        C = ode_strength_of_connection(A, B, **kwargs)
    elif fn == 'energy_based':
        C = energy_based_strength_of_connection(A, **kwargs)
    elif fn is None:
        C = A
    else:
        raise ValueError('unrecognized strength of connection method: %s' % str(fn))

    # In SA, strength represents "distance", so we take magnitude of complex values
    if C.dtype == complex:
        C.data = numpy.abs(C.data)

    ##
    # aggregation
    fn, kwargs = unpack_arg(aggregate)
    if fn == 'standard':
        AggOp = standard_aggregation(C, **kwargs)
    elif fn == 'lloyd':
        AggOp = lloyd_aggregation(C, **kwargs)
    elif fn == 'predefined':
        AggOp = aggregate[1][len(levels) - 1]
    else:
        raise ValueError('unrecognized aggregation method %s' % str(fn))

    ##
    # tentative prolongator
    T,B = fit_candidates(AggOp,B)

    ##
    # tentative prolongator smoother
    fn, kwargs = unpack_arg(smooth)
    if fn == 'jacobi':
        P = jacobi_prolongation_smoother(A, T, **kwargs)
    elif fn == 'richardson':
        P = richardson_prolongation_smoother(A, T, **kwargs)
    elif fn == 'energy':
        #from scipy import conjugate
        #R = energy_prolongation_smoother(A.H.asformat(A.format), T, C, conjugate(B), **kwargs).H
        P = energy_prolongation_smoother(A, T, C, B, **kwargs)
    elif fn == 'kaczmarz_richardson':
        P = kaczmarz_richardson_prolongation_smoother(A, T, **kwargs)
    elif fn == 'kaczmarz_jacobi':
        P = kaczmarz_jacobi_prolongation_smoother(A, T, **kwargs)
    elif fn is None:
        P = T
    else:
        raise ValueError('unrecognized prolongation smoother method %s' % str(fn))
   
    ##
    # Choice of R reflects A's structure
    symmetry = A.symmetry
    #if fn != 'energy':
    if True:
        if symmetry == 'hermitian':
            R = P.H
        elif symmetry == 'symmetric':
            R = P.T

    levels[-1].C     = C       # strength of connection matrix
    levels[-1].AggOp = AggOp   # aggregation operator
    levels[-1].T     = T       # tentative prolongator
    levels[-1].P     = P       # smoothed prolongator
    levels[-1].R     = R       # restriction operator 

    A = R * A * P              # galerkin operator
    A.symmetry = symmetry
    
    levels.append( multilevel_solver.level() )
    levels[-1].A = A
    levels[-1].B = B

## unused, but useful
def sa_filtered_matrix(A,theta):
    """The filtered matrix is obtained from A by lumping all weak off-diagonal
    entries onto the diagonal.  Weak off-diagonals are determined by
    the standard strength of connection measure using the parameter theta.

    In the case theta = 0.0, (i.e. no weak connections) A is returned.
    """

    if theta == 0:
        return A

    if isspmatrix_csr(A): 
        #TODO rework this
        raise NotImplementedError('blocks not handled yet')
        Sp,Sj,Sx = amg_core.symmetric_strength_of_connection(A.shape[0],theta,A.indptr,A.indices,A.data)
        return csr_matrix((Sx,Sj,Sp),shape=A.shape)
    elif ispmatrix_bsr(A):
        raise NotImplementedError('blocks not handled yet')
    else:
        return sa_filtered_matrix(csr_matrix(A),theta)
##            #TODO subtract weak blocks from diagonal blocks?
##            num_dofs   = A.shape[0]
##            num_blocks = blocks.max() + 1
##
##            if num_dofs != len(blocks):
##                raise ValueError,'improper block specification'
##
##            # for non-scalar problems, use pre-defined blocks in aggregation
##            # the strength of connection matrix is based on the 1-norms of the blocks
##
##            B  = csr_matrix((numpy.ones(num_dofs),blocks,numpy.arange(num_dofs + 1)),shape=(num_dofs,num_blocks))
##            Bt = B.T.tocsr()
##
##            #1-norms of blocks entries of A
##            Block_A = Bt * csr_matrix((numpy.abs(A.data),A.indices,A.indptr),shape=A.shape) * B
##
##            S = symmetric_strength_of_connection(Block_A,theta)
##            S.data[:] = 1
##
##            Mask = B * S * Bt
##
##            A_strong = A ** Mask
##            #A_weak   = A - A_strong
##            A_filtered = A_strong
    return A_filtered
