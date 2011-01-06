"""Classical AMG (Ruge-Stuben AMG)"""

__docformat__ = "restructuredtext en"

from scipy.sparse import csr_matrix, isspmatrix_csr, isspmatrix_bsr, eye

from pyamg.multilevel import multilevel_solver
from pyamg.relaxation.smoothing import change_smoothers
from pyamg.strength import classical_strength_of_connection, \
                           symmetric_strength_of_connection, \
                           ode_strength_of_connection

from interpolate import direct_interpolation

__all__ = ['ruge_stuben_solver']

def ruge_stuben_solver(A, 
                       strength=('classical',{'theta':0.25}), 
                       CF='RS', 
                       presmoother=('gauss_seidel',{'sweep':'symmetric'}),
                       postsmoother=('gauss_seidel',{'sweep':'symmetric'}),
                       max_levels=10, max_coarse=500, keep=False, **kwargs):
    """Create a multilevel solver using Classical AMG (Ruge-Stuben AMG)

    Parameters
    ----------
    A : csr_matrix
        Square matrix in CSR format
    strength : ['symmetric', 'classical', 'ode', None]
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
    smoothed_aggregation_solver, multilevel_solver

    """

    levels = [ multilevel_solver.level() ]
    
    # convert A to csr
    if not isspmatrix_csr(A):
        try:
            A = csr_matrix(A)
            print 'Implicit conversion of A to CSR in pyamg.ruge_stuben_solver'
        except:
            raise TypeError('Argument A must have type csr_matrix, or be convertible to csr_matrix')
    # preprocess A
    A = A.asfptype()
    if A.shape[0] != A.shape[1]:
        raise ValueError('expected square matrix')

    levels[-1].A = A
    
    while len(levels) < max_levels  and levels[-1].A.shape[0] > max_coarse:
        extend_hierarchy(levels, strength, CF, keep)

    ml = multilevel_solver(levels, **kwargs)
    change_smoothers(ml, presmoother, postsmoother)
    return ml

# internal function
def extend_hierarchy(levels, strength, CF, keep):
    """ helper function for local methods """
    def unpack_arg(v):
        if isinstance(v,tuple):
            return v[0],v[1]
        else:
            return v,{}
    
    A = levels[-1].A

    # strength of connection
    fn, kwargs = unpack_arg(strength)
    if fn == 'symmetric':
        C = symmetric_strength_of_connection(A,**kwargs)
    elif fn == 'classical':
        C = classical_strength_of_connection(A,**kwargs)
    elif fn == 'ode':
        raise NotImplementedError('ode method not supported for Classical AMG')
    elif fn is None:
        C = A
    else:
        raise ValueError('unrecognized strength of connection method: %s' % fn)


    if CF in [ 'RS', 'PMIS', 'PMISc', 'CLJP', 'CLJPc']:
        import split
        splitting = getattr(split, CF)(C)
    else:
        raise ValueError('unknown C/F splitting method (%s)' % CF)

    P = direct_interpolation(A, C, splitting)


    R = P.T.tocsr()

    if keep:
        levels[-1].C = C                  # strength of connection matrix
        levels[-1].splitting = splitting  # C/F splitting

    levels[-1].P = P                  # prolongation operator
    levels[-1].R = R                  # restriction operator

    levels.append( multilevel_solver.level() )

    A = R * A * P                     #galerkin operator
    levels[-1].A = A

