"""Support for aggregation-based AMG"""

__docformat__ = "restructuredtext en"

import numpy
import types
from scipy.sparse import csr_matrix, isspmatrix_csr, isspmatrix_bsr, eye

from pyamg import relaxation
from pyamg import amg_core
from pyamg.multilevel import multilevel_solver
from pyamg.relaxation.smoothing import change_smoothers
from pyamg.util.utils import symmetric_rescaling_sa, diag_sparse, amalgamate, \
                             relaxation_as_linear_operator
from pyamg.strength import classical_strength_of_connection, \
        symmetric_strength_of_connection, ode_strength_of_connection, \
        energy_based_strength_of_connection, distance_strength_of_connection
from aggregate import standard_aggregation, naive_aggregation, lloyd_aggregation, \
                      anisotropic_aggregation, anisotropic_postprocessing
from tentative import fit_candidates
from smooth import jacobi_prolongation_smoother, richardson_prolongation_smoother, \
        energy_prolongation_smoother

__all__ = ['smoothed_aggregation_solver']

def nPDEs(levels):
    # Helper Function:
    # Return the number of PDEs (i.e. blocksize) at the coarsest level
    
    if isspmatrix_bsr(levels[-1].A):
        return levels[-1].A.blocksize[0]
    else:
        # csr matrices correspond to 1 PDE
        return 1

def preprocess_Bimprove(Bimprove, A, max_levels):
    # Helper function for smoothed_aggregation_solver.  Upon return,
    # Bimprove[i] is length max_levels and defines the Bimprove routine 
    # for level i.
    
    if Bimprove == 'default':
        if A.symmetry == 'hermitian' or A.symmetry == 'symmetric':
            Bimprove = [('block_gauss_seidel', {'sweep':'symmetric', 'iterations':4}), None]
        else:    
            Bimprove = [('gauss_seidel_nr', {'sweep':'symmetric', 'iterations':4}), None]
    elif Bimprove == None:
        Bimprove = [None]

    if not isinstance(Bimprove, list):
        raise ValueError("Bimprove must be a list")
    elif len(Bimprove) < max_levels:
            Bimprove.extend([Bimprove[-1] for i in range(max_levels-len(Bimprove)) ])

    return Bimprove


def preprocess_str_or_agg(scheme, max_levels, max_coarse):
    # Helper function for smoothed_aggregation_solver that preprocesses
    # strength of connection and aggregation parameters from the user.  Upon
    # return, scheme[i] is length max_levels and defines the scheme or
    # aggregation routine for level i.

    if isinstance(scheme, tuple):
        if scheme[0] == 'predefined':
            scheme = [scheme] 
            max_levels = 2
            max_coarse = 0
        else:
            scheme = [scheme for i in range(max_levels-1)]
    
    elif isinstance(scheme,str):
        if scheme == 'predefined':
            raise ValueError('predefined scheme requires a user-provided ' +\
                             'CSR matrix representing strength or aggregation' +\
                             'i.e., (\'predefined\', {\'C\' : CSR_MAT}).')
        else:
            scheme = [scheme for i in range(max_levels-1)]
    
    elif isinstance(scheme, list):
        if isinstance(scheme[-1], tuple) and (scheme[-1][0] == 'predefined'): 
            # scheme is a list that ends with a predefined operator
            max_levels = len(scheme) + 1
            max_coarse = 0
        else:
            # scheme a list that __doesn't__ end with 'predefined'
            if len(scheme) < max_levels-1:
                scheme.extend([scheme[-1] for i in range(max_levels-len(scheme)-1) ])

    elif scheme==None:
        scheme=[(None,{}) for i in range(max_levels-1)]
    else:
        raise ValueError('invalid scheme')

    return max_levels, max_coarse, scheme


def preprocess_smooth(smooth, max_levels):
    # Helper function for smoothed_aggregation_solver.  Upon return,
    # smooth[i] is length max_levels and defines the smooth routine 
    # for level i.
    
    if isinstance(smooth, tuple) or isinstance(smooth,str):
        smooth = [smooth for i in range(max_levels)]
    elif isinstance(smooth, list):
        if len(smooth) < max_levels:
            smooth.extend([smooth[-1] for i in range(max_levels-len(smooth)) ])
    elif smooth==None:
        smooth=[(None,{}) for i in range(max_levels)]
    
    return smooth


def smoothed_aggregation_solver(A, B=None, BH=None,
        symmetry='hermitian', strength='symmetric', 
        aggregate='standard', smooth=('jacobi', {'omega': 4.0/3.0}),
        presmoother=('block_gauss_seidel',{'sweep':'symmetric'}),
        postsmoother=('block_gauss_seidel',{'sweep':'symmetric'}),
        Bimprove='default', max_levels = 10, max_coarse = 500, **kwargs):
    """
    Create a multilevel solver using Smoothed Aggregation (SA)

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
        Note that for the strictly real case, symmetric and hermitian are the same
        Note that this flag does not denote definiteness of the operator.
    strength : ['symmetric', 'classical', 'ode', ('predefined', {'C' : csr_matrix}), None]
        Method used to determine the strength of connection between unknowns of
        the linear system.  Method-specific parameters may be passed in using a
        tuple, e.g. strength=('symmetric',{'theta' : 0.25 }). If strength=None,
        all nonzero entries of the matrix are considered strong.  
            See notes below for varying this parameter on a per level basis.  Also,
        see notes below for using a predefined strength matrix on each level.
    aggregate : ['standard', 'lloyd', 'naive', ('predefined', {'AggOp' : csr_matrix})]
        Method used to aggregate nodes.  See notes below for varying this
        parameter on a per level basis.  Also, see notes below for using a
        predefined aggregation on each level.
    smooth : ['jacobi', 'richardson', 'energy', None]
        Method used to smooth the tentative prolongator.  Method-specific
        parameters may be passed in using a tuple, e.g.  smooth=
        ('jacobi',{'filter' : True }).  See notes below for varying this
        parameter on a per level basis.
    presmoother : {tuple, string, list} : default ('block_gauss_seidel', {'sweep':'symmetric'})
        Defines the presmoother for the multilevel cycling.  The default block
        Gauss-Seidel option defaults to point-wise Gauss-Seidel, if the matrix
        is CSR or is a BSR matrix with blocksize of 1.  See notes below for
        varying this parameter on a per level basis.
    postsmoother : {tuple, string, list}
        Same as presmoother, except defines the postsmoother.
    Bimprove : {list} : default [('block_gauss_seidel', {'sweep':'symmetric'}), None]
        The ith entry defines the method used to improve the candidates B on
        level i.  If the list is shorter than max_levels, then the last entry
        will define the method for all levels lower.
            The list elements are relaxation descriptors of the form used for
        presmoother and postsmoother.  A value of None implies no action on B.
    max_levels : {integer} : default 10
        Maximum number of levels to be used in the multilevel solver.
    max_coarse : {integer} : default 500
        Maximum number of variables permitted on the coarse grid. 

    Other Parameters
    ----------------
    cycle_type : ['V','W','F']
        Structrure of multigrid cycle
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
    
    - The parameters smooth, strength, aggregate, presmoother, postsmoother can
      be varied on a per level basis.  For different methods on different
      levels, use a list as input so that the ith entry defines the method at
      the ith level.  If there are more levels in the hierarchy than list
      entries, the last entry will define the method for all levels lower.
      
      Examples are:
      smooth=[('jacobi', {'omega':1.0}), None, 'jacobi']
      presmoother=[('block_gauss_seidel', {'sweep':symmetric}), 'sor']
      aggregate=['standard', 'naive']
      strength=[('symmetric', {'theta':0.25}), ('symmetric',{'theta':0.08})]

    - Predefined strength of connection and aggregation schemes can be
      specified.  These options are best used together, but aggregation can be
      predefined while strength of connection is not.

      For predefined strength of connection, use a list consisting of tuples of
      the form ('predefined', {'C' : C0}), where C0 is a csr_matrix and each
      degree-of-freedom in C0 represents a supernode.  For instance to
      predefine a three-level hierarchy, use [('predefined', {'C' : C0}),
      ('predefined', {'C' : C1}) ].
      
      Similarly for predefined aggregation, use a list of tuples.  For instance
      to predefine a three-level hierarchy, use [('predefined', {'AggOp' :
      Agg0}), ('predefined', {'AggOp' : Agg1}) ], where the dimensions of A,
      Agg0 and Agg1 are compatible, i.e.  Agg0.shape[1] == A.shape[0] and
      Agg1.shape[1] == Agg0.shape[0].  Each AggOp is a csr_matrix.


    Examples
    --------
    >>> from pyamg import smoothed_aggregation_solver
    >>> from pyamg.gallery import poisson
    >>> from scipy.sparse.linalg import cg
    >>> import numpy
    >>> A = poisson((100,100), format='csr')           # matrix
    >>> b = numpy.ones((A.shape[0]))                   # RHS
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
        try:
            A = csr_matrix(A)
            print 'Implicit conversion of A to CSR in pyamg.smoothed_aggregation_solver'
        except:
            raise TypeError('Argument A must have type csr_matrix or bsr_matrix,\
                             or be convertible to csr_matrix')

    A = A.asfptype()
    
    if (symmetry != 'symmetric') and (symmetry != 'hermitian') and (symmetry != 'nonsymmetric'):
        raise ValueError('expected \'symmetric\', \'nonsymmetric\' or \'hermitian\' for the symmetry parameter ')
    A.symmetry = symmetry

    if A.shape[0] != A.shape[1]:
        raise ValueError('expected square matrix')
    ##
    # Right near nullspace candidates
    if B is None:
        B = numpy.ones((A.shape[0],1), dtype=A.dtype) # use constant vector
    else:
        B = numpy.asarray(B, dtype=A.dtype)
    
    ##
    # Left near nullspace candidates
    if A.symmetry == 'nonsymmetric':
        if BH is None:
            BH = B.copy()
        else:
            BH = numpy.asarray(BH, dtype=A.dtype)

    ##
    # Preprocess parameters
    max_levels, max_coarse, strength = preprocess_str_or_agg(strength, max_levels, max_coarse)
    max_levels, max_coarse, aggregate = preprocess_str_or_agg(aggregate, max_levels, max_coarse)
    Bimprove = preprocess_Bimprove(Bimprove, A, max_levels)
    smooth = preprocess_smooth(smooth, max_levels)
   
    ##
    # Construct multilevel structure
    levels = []
    levels.append( multilevel_solver.level() )
    levels[-1].A = A          # matrix
   
    ##
    # Append near null-space candidates
    levels[-1].B = B          # right candidates
    if A.symmetry == 'nonsymmetric':
        levels[-1].BH = BH    # left candidates
    
    while len(levels) < max_levels and levels[-1].A.shape[0]/nPDEs(levels) > max_coarse:
        extend_hierarchy(levels, strength, aggregate, smooth, Bimprove)
    
    ml = multilevel_solver(levels, **kwargs)
    change_smoothers(ml, presmoother, postsmoother)
    return ml

def extend_hierarchy(levels, strength, aggregate, smooth, Bimprove):
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
    if A.symmetry == "nonsymmetric":
        AH = A.H.asformat(A.format)
        BH = levels[-1].BH
 
    ##
    # Begin constructing next level
    fn, kwargs = unpack_arg(strength[len(levels)-1])
    if fn == 'symmetric':
        C = symmetric_strength_of_connection(A, **kwargs)
        C = C + eye(C.shape[0], C.shape[1], format='csr')   # Diagonal must be nonzero
    elif fn == 'classical':
        C = classical_strength_of_connection(A, **kwargs)
        C = C + eye(C.shape[0], C.shape[1], format='csr')   # Diagonal must be nonzero
        if isspmatrix_bsr(A):
            C = amalgamate(C, A.blocksize[0])
    elif fn == 'distance':
        C = distance_strength_of_connection(A, **kwargs)
    elif fn == 'ode':
        C = ode_strength_of_connection(A, B, **kwargs)
    elif fn == 'energy_based':
        C = energy_based_strength_of_connection(A, **kwargs)
    elif fn == 'predefined':
        C = kwargs['C'].tocsr()
    elif fn is None:
        C = A.tocsr()
    else:
        raise ValueError('unrecognized strength of connection method: %s' % str(fn))
    
    # In SA, strength represents "distance", so we take magnitude of complex values
    if C.dtype == complex:
        C.data = numpy.abs(C.data)
    
    # Create a unified strength framework so that large values represent strong
    # connections and small values represent weak connections
    if (fn == 'ode') or (fn == 'distance') or (fn == 'energy_based'):
        C.data = 1.0/C.data

    ##
    # aggregation
    fn, kwargs = unpack_arg(aggregate[len(levels)-1])
    if fn == 'standard':
        AggOp = standard_aggregation(C, **kwargs)
    elif fn == 'naive':
        AggOp = naive_aggregation(C, **kwargs)
    elif fn == 'lloyd':
        AggOp = lloyd_aggregation(C, **kwargs)
    elif fn == 'predefined':
        AggOp = kwargs['AggOp'].tocsr()
    else:
        raise ValueError('unrecognized aggregation method %s' % str(fn))
    
    ##
    # Improve near null-sapce candidates (important to place after the call to
    # ode_strength_of_connection)
    if Bimprove[len(levels)-1] is not None:
        b = numpy.zeros((A.shape[0],1), dtype=A.dtype)
        B = relaxation_as_linear_operator(Bimprove[len(levels)-1], A, b) * B
        levels[-1].B = B
        if A.symmetry == "nonsymmetric":
            BH = relaxation_as_linear_operator(Bimprove[len(levels)-1], AH, b) * BH 
            levels[-1].BH = BH

    ##
    # tentative prolongator
    T,B = fit_candidates(AggOp,B)
    if A.symmetry == "nonsymmetric":
        TH,BH = fit_candidates(AggOp,BH)

    ##
    # tentative prolongator smoother
    fn, kwargs = unpack_arg(smooth[len(levels)-1])
    if fn == 'jacobi':
        P = jacobi_prolongation_smoother(A, T, C, B, **kwargs)
    elif fn == 'richardson':
        P = richardson_prolongation_smoother(A, T, **kwargs)
    elif fn == 'energy':
        P,B = energy_prolongation_smoother(A, T, C, B, **kwargs)
    elif fn is None:
        P = T
    else:
        raise ValueError('unrecognized prolongation smoother method %s' % str(fn))
   
    ##
    # Choice of R reflects A's structure
    symmetry = A.symmetry
    if symmetry == 'hermitian':
        R = P.H
    elif symmetry == 'symmetric':
        R = P.T
    elif symmetry == 'nonsymmetric':
        fn, kwargs = unpack_arg(smooth[len(levels)-1])
        if fn == 'jacobi':
            R = jacobi_prolongation_smoother(AH, TH, C, BH, **kwargs).H
        elif fn == 'richardson':
            R = richardson_prolongation_smoother(AH, TH, **kwargs).H
        elif fn == 'energy':
            R,BH = energy_prolongation_smoother(AH, TH, C, BH, **kwargs)
            R = R.H
        elif fn is None:
            R = T.H
        else:
            raise ValueError('unrecognized prolongation smoother method %s' % str(fn))


    levels[-1].C     = C       # strength of connection matrix
    levels[-1].AggOp = AggOp   # aggregation operator
    levels[-1].T     = T       # tentative prolongator
    levels[-1].P     = P       # smoothed prolongator
    levels[-1].R     = R       # restriction operator 

    A = R * A * P              # galerkin operator
    A.symmetry = symmetry
    
    levels.append( multilevel_solver.level() )
    levels[-1].A = A
       
    levels[-1].B = B           # right near null-space candidates
    if A.symmetry == "nonsymmetric":
        levels[-1].BH = BH     # left near null-space candidates

