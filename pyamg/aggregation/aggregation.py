"""Support for aggregation-based AMG"""

__docformat__ = "restructuredtext en"

from numpy import array, arange, ones, zeros, sqrt, asarray, \
        empty, empty_like, diff

from scipy.sparse import csr_matrix, coo_matrix, \
        isspmatrix_csr, bsr_matrix, isspmatrix_bsr

from pyamg import multigridtools
from pyamg.multilevel import multilevel_solver
from pyamg.utils import symmetric_rescaling, diag_sparse, scale_columns

from pyamg.strength import *
from aggregate import *
from tentative import fit_candidates
from smooth import *

__all__ = ['smoothed_aggregation_solver']


def smoothed_aggregation_solver(A, B=None, 
        strength='symmetric', 
        aggregate='standard', 
        smooth=('jacobi', {'omega': 4.0/3.0}),
        max_levels = 10, max_coarse = 500, **kwargs):
    """Create a multilevel solver using Smoothed Aggregation (SA)

    Setup Parameters
    ----------------
    A : {csr_matrix, bsr_matrix}
        Sparse NxN matrix in CSR or BSR format
    B : {None, array_like}
        Near-nullspace candidates stored in the columns of an NxK array.
        The default value B=None is equivalent to B=ones((N,1))
    strength : ['symmetric', 'classical', 'ode', None]
        Method used to determine the strength of connection between unknowns
        of the linear system.  Method-specific parameters may be passed in
        using a tuple, e.g. strength=('symmetric',{'theta' : 0.25 }). If
        strength=None, all nonzero entries of the matrix are considered strong.
    aggregate : ['standard', 'lloyd', ('predefined',[csr_matrix, ...])]
        Method used to aggregate nodes.  A predefined aggregation is specified 
        with a sequence of csr_matrices that represent the aggregation operators
        on each level of the hierarchy.
    smooth : ['jacobi', 'chebyshev', 'MLS', 'energy', None]
        Method used to smoother used to smooth the tentative prolongator.
    max_levels : {integer} : default 10
        Maximum number of levels to be used in the multilevel solver.
    max_coarse : {integer} : default 500
        Maximum number of variables permitted on the coarse grid.
    
    TODO ADD PREPROCESSES            
    
    Cycle Parameters
    ----------------
    cycle_type : ['V','W','F']
        Structrure of multigrid cycle
    presmoother  : ['gauss_seidel', 'jacobi', ... ]
        Premoother used during multigrid cycling
    postsmoother : ['gauss_seidel', 'jacobi', ... ]
        Postmoother used during multigrid cycling
    coarse_solver : ['splu','lu', ... ]
        Solver used at the coarsest level of the MG hierarchy 

    Cycle parameters are passed through as arguments to multilevel_solver.
    Refer to pyamg.multilevel_solver for additional documentation.
    

    Notes
    -----
    TODO Distinguish betwenn setup and cycle parameters
    TODO describe sequence of operations on each level
    preprocess -> strength -> aggregate -> tentative -> smooth



    Unused Parameters
    -----------------
    symmetric: {boolean} : default True
        True if A is symmetric, False otherwise
    rescale: {boolean} : default True
        If True, symmetrically rescale A by the diagonal
        i.e. A -> D * A * D,  where D is diag(A)^-0.5
    aggregation: {sequence of csr_matrix objects}
        List of csr_matrix objects that describe a user-defined
        multilevel aggregation of the degrees of freedom.
        For instance [ Agg0, Agg1 ] defines a three-level hierarchy
        where the dimensions of A, Agg0 and Agg1 are compatible, i.e.
        Agg0.shape[1] == A.shape[0] and Agg1.shape[1] == Agg0.shape[0].
  


    Example
    -------
        TODO
        TODO show aggregate=('predefined',[AggOp1,AggOp2])

    References
    ----------

        Petr Vanek, Jan Mandel and Marian Brezina
        "Algebraic Multigrid by Smoothed Aggregation for Second and Fourth Order Elliptic Problems",
        http://citeseer.ist.psu.edu/vanek96algebraic.html

    """

    A = A.asfptype()

    if not (isspmatrix_csr(A) or isspmatrix_bsr(A)):
        raise TypeError('argument A must have type csr_matrix or bsr_matrix')

    if A.shape[0] != A.shape[1]:
        raise ValueError('expected square matrix')

    if B is None:
        B = ones((A.shape[0],1), dtype=A.dtype) # use constant vector
    else:
        B = asarray(B, dtype=A.dtype)
    
    pre,post = None,None   #preprocess/postprocess

    #if rescale:
    #    D_sqrt,D_sqrt_inv,A = symmetric_rescaling(A)
    #    D_sqrt,D_sqrt_inv = diag_sparse(D_sqrt),diag_sparse(D_sqrt_inv)

    #    B = D_sqrt * B  #scale candidates
    #    def pre(x,b):
    #        return D_sqrt*x,D_sqrt_inv*b
    #    def post(x):
    #        return D_sqrt_inv*x

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
        
    return multilevel_solver(levels, **kwargs)


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
        Sp,Sj,Sx = multigridtools.symmetric_strength_of_connection(A.shape[0],theta,A.indptr,A.indices,A.data)
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
##            B  = csr_matrix((ones(num_dofs),blocks,arange(num_dofs + 1)),shape=(num_dofs,num_blocks))
##            Bt = B.T.tocsr()
##
##            #1-norms of blocks entries of A
##            Block_A = Bt * csr_matrix((abs(A.data),A.indices,A.indptr),shape=A.shape) * B
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





def extend_hierarchy(levels, strength, aggregate, smooth):
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
    elif fn is None:
        C = A
    else:
        raise ValueError('unrecognized strength of connection method: %s' % str(fn))

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
    elif fn == 'energy':
        P = energy_prolongation_smoother(A, T, C, B, **kwargs)
    elif fn is None:
        P = T
    else:
        raise ValueError('unrecognized prolongation smoother method %s' % str(fn))
   

    
    R = P.T.asformat(P.format)

    levels[-1].C     = C       # strength of connection matrix
    levels[-1].AggOp = AggOp   # aggregation operator
    levels[-1].T     = T       # tentative prolongator
    levels[-1].P     = P       # smoothed prolongator
    levels[-1].R     = R       # restriction operator 

    A = R * A * P              # galerkin operator
    
    levels.append( multilevel_solver.level() )
    levels[-1].A = A
    levels[-1].B = B






