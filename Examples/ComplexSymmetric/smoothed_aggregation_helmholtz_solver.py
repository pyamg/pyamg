"""Support for aggregation-based AMG"""

__docformat__ = "restructuredtext en"

import numpy
import scipy
from scipy.sparse import isspmatrix_csr, isspmatrix_bsr

from pyamg.multilevel import multilevel_solver
from pyamg.util.utils import relaxation_as_linear_operator
from pyamg.relaxation.smoothing import change_smoothers
from pyamg.strength import symmetric_strength_of_connection, evolution_strength_of_connection, \
                           distance_strength_of_connection
from pyamg.aggregation.aggregation import extend_hierarchy, preprocess_Bimprove, \
                                        preprocess_str_or_agg, preprocess_smooth
from pyamg.aggregation.aggregate import standard_aggregation, lloyd_aggregation
from pyamg.aggregation.tentative import fit_candidates
from pyamg.aggregation.smooth import jacobi_prolongation_smoother, \
                    richardson_prolongation_smoother, energy_prolongation_smoother

 
__all__ = ['smoothed_aggregation_helmholtz_solver', 'planewaves']

def planewaves(X, Y, omega=1.0, angles=[0.0]):
    """
    Generate plane waves for use in SA applied to Helmholtz problems

    Parameters
    ----------

    X,Y : {array}
        Coordinate vectors
    omega : {float}
        Helmholtz wave number, Laplace(u) + omega^2 u = f
    angles : {list}
        List of angles in [0, 2 pi] from which to generate planewaves
    
    Returns
    -------
    Array of planewaves
    
    """   

    L = 2*len(angles)
    dimen = max(X.shape)
    W = numpy.zeros((L, dimen),dtype=complex)
    
    if L == 0:
        W = W.T.copy()
        return W
    
    X = numpy.ravel(X)
    Y = numpy.ravel(Y)

    #Set other columns to plane waves
    counter = 0
    for angle in angles:
        K = (omega*numpy.cos(angle), omega*numpy.sin(angle))
        wave = numpy.exp(0 + 1.0j*K[0]*X + 1.0j*K[1]*Y)

        W[counter,:] = numpy.real(wave)
        W[counter+1,:] = numpy.imag(wave)
        counter += 2
    
    # write W row-wise for efficiency
    W = W.T.copy()
    return W

def preprocess_planewaves(planewaves, max_levels):
    # Helper function for smoothed_aggregation_solver.   
    # Will extend planewaves to a length max_levels list, repeating
    # the final element of planewaves if necessary.

    if planewaves == None:
        planewaves = [None]
    if not isinstance(planewaves, list):
        raise ValueError("planewaves must be a list")
    elif len(planewaves) < max_levels:
            planewaves.extend([planewaves[-1] for i in range(max_levels-len(planewaves)) ])

    return planewaves


def unpack_arg(v):
    if isinstance(v,tuple):
        return v[0],v[1]
    else:
        return v,{}

def smoothed_aggregation_helmholtz_solver(A, planewaves, use_constant=(True, {'last_level':0}), 
        symmetry='symmetric', strength='symmetric', aggregate='standard',
        smooth=('energy', {'krylov': 'gmres'}),
        presmoother=('gauss_seidel_nr',{'sweep':'symmetric'}),
        postsmoother=('gauss_seidel_nr',{'sweep':'symmetric'}),
        Bimprove='default', max_levels = 10, max_coarse = 100, **kwargs):
    
    """
    Create a multilevel solver using Smoothed Aggregation (SA) for a 2D Helmholtz operator

    Parameters
    ----------
    A : {csr_matrix, bsr_matrix}
        Sparse NxN matrix in CSR or BSR format
    planewaves : { list }
        [pw_0, pw_1, ..., pw_n], where the k-th tuple pw_k is of the form (fn,
        args).  fn is a callable and args is a dictionary of arguments for fn.
        This k-th tuple is used to define any new planewaves (i.e., new coarse
        grid basis functions) to be appended to the existing B_k at that level. 
            The function fn must return functions defined on the finest level, 
        i.e., a collection of vector(s) of length A.shape[0].  These vectors
        are then restricted to the appropriate level, where they enrich the 
        coarse space.
            Instead of a tuple, None can be used to stipulate no introduction
        of planewaves at that level.  If len(planewaves) < max_levels, the 
        last entry is used to define coarser level planewaves.
    use_constant : {tuple}
        Tuple of the form (bool, {'last_level':int}).  The boolean denotes 
        whether to introduce the constant in B at level 0.  'last_level' denotes
        the final level to use the constant in B.  That is, if 'last_level' is 1,
        then the vector in B corresponding to the constant on level 0 is dropped 
        from B at level 2.
            This is important, because using constant based interpolation beyond
        the Nyquist rate will result in poor solver performance.
    symmetry : {string}
        'symmetric' refers to both real and complex symmetric
        'hermitian' refers to both complex Hermitian and real Hermitian
        'nonsymmetric' i.e. nonsymmetric in a hermitian sense
        Note that for the strictly real case, symmetric and hermitian are the same
        Note that this flag does not denote definiteness of the operator.
    strength : ['symmetric', 'classical', 'evolution', ('predefined', {'C' : csr_matrix}), None]
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
    coarse_solver : ['splu','lu', ... ]
        Solver used at the coarsest level of the MG hierarchy 

    Returns
    -------
    ml : multilevel_solver
        Multigrid hierarchy of matrices and prolongation operators

    See Also
    --------
    multilevel_solver, smoothed_aggregation_solver

    Notes
    -----
    - The additional parameters are passed through as arguments to
      multilevel_solver.  Refer to pyamg.multilevel_solver for additional
      documentation.

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
    >>> from pyamg import smoothed_aggregation_helmholtz_solver, poisson
    >>> from scipy.sparse.linalg import cg
    >>> from scipy import rand
    >>> A = poisson((100,100), format='csr')           # matrix
    >>> b = rand(A.shape[0])                           # random RHS
    >>> ml = smoothed_aggregation_solver(A)            # AMG solver
    >>> M = ml.aspreconditioner(cycle='V')             # preconditioner
    >>> x,info = cg(A, b, tol=1e-8, maxiter=30, M=M)   # solve with CG

    References
    ----------
    .. [1] L. N. Olson and J. B. Schroder. Smoothed Aggregation for Helmholtz
    Problems. Numerical Linear Algebra with Applications.  pp. 361--386.  17
    (2010).

    """
    if not (isspmatrix_csr(A) or isspmatrix_bsr(A)):
        raise TypeError('argument A must have type csr_matrix or bsr_matrix')

    A = A.asfptype()
    
    if (symmetry != 'symmetric') and (symmetry != 'hermitian') and (symmetry != 'nonsymmetric'):
        raise ValueError('expected \'symmetric\', \'nonsymmetric\' or \'hermitian\' for the symmetry parameter ')
    A.symmetry = symmetry

    if A.shape[0] != A.shape[1]:
        raise ValueError('expected square matrix')
    
    ##
    # Preprocess and extend planewaves to length max_levels
    planewaves = preprocess_planewaves(planewaves, max_levels)
    # Check that the user has defined functions for B at each level
    use_const, args = unpack_arg(use_constant)
    first_planewave_level = -1
    for pw in planewaves:
        first_planewave_level += 1
        if pw is not None:
            break
    ##    
    if (use_const == False) and (planewaves[0] == None):
        raise ValueError('No functions defined for B on the finest level, ' + \
              'either use_constant must be true, or planewaves must be defined for level 0')
    elif (use_const == True) and (args['last_level'] < first_planewave_level-1):
        raise ValueError('Some levels have no function(s) defined for B.  ' + \
                         'Change use_constant and/or planewave arguments.')
        
    ##
    # Preprocess parameters
    max_levels, max_coarse, strength = preprocess_str_or_agg(strength, max_levels, max_coarse)
    max_levels, max_coarse, aggregate = preprocess_str_or_agg(aggregate, max_levels, max_coarse)
    Bimprove = preprocess_Bimprove(Bimprove, A, max_levels)
    smooth = preprocess_smooth(smooth, max_levels)

    ##
    # Start first level
    levels = []
    levels.append( multilevel_solver.level() )
    levels[-1].A = A                            # matrix
    levels[-1].B = numpy.zeros((A.shape[0],0))  # place-holder for near-nullspace candidates

    zeros_0 = numpy.zeros((levels[0].A.shape[0],), dtype=A.dtype)
    while len(levels) < max_levels and levels[-1].A.shape[0] > max_coarse:
        A = levels[0].A
        A_l = levels[-1].A
        zeros_l = numpy.zeros((levels[-1].A.shape[0],), dtype=A.dtype)

        ##
        # Generate additions to n-th level candidates
        if planewaves[len(levels)-1] != None:
            fn, args = unpack_arg(planewaves[len(levels)-1])
            Bcoarse2 = numpy.array(fn(**args))

            ##
            # As in alpha-SA, relax the candidates before restriction
            if Bimprove[0] is not None:
                Bcoarse2 = relaxation_as_linear_operator(Bimprove[0], A, zeros_0)*Bcoarse2
            
            ##
            # Restrict Bcoarse2 to current level
            for i in range(len(levels)-1):
                Bcoarse2 = levels[i].R*Bcoarse2
            # relax after restriction
            if Bimprove[len(levels)-1] is not None:
                Bcoarse2 =relaxation_as_linear_operator(Bimprove[len(levels)-1],A_l,zeros_l)*Bcoarse2
        else:
            Bcoarse2 = numpy.zeros((A_l.shape[0],0),dtype=A.dtype)

        ##
        # Deal with the use of constant in interpolation
        use_const, args = unpack_arg(use_constant)
        if use_const and len(levels) == 1:
            # If level 0, and the constant is to be used in interpolation
           levels[0].B = numpy.hstack( (numpy.ones((A.shape[0],1), dtype=A.dtype), Bcoarse2) )
        elif use_const and args['last_level'] == len(levels)-2: 
            # If the previous level was the last level to use the constant, then remove the
            # coarse grid function based on the constant from B
            levels[-1].B = numpy.hstack( (levels[-1].B[:,1:], Bcoarse2) )
        else:
            levels[-1].B = numpy.hstack((levels[-1].B, Bcoarse2))
        
        ##
        # Create and Append new level
        extend_hierarchy(levels, strength, aggregate, smooth, [None for i in range(max_levels)] )
    
    ml = multilevel_solver(levels, **kwargs)
    change_smoothers(ml, presmoother, postsmoother)
    return ml


