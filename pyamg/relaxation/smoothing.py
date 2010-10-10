"""Method to create pre- and post-smoothers on the levels of a multilevel_solver
"""

import numpy
import scipy
import relaxation
from chebyshev import chebyshev_polynomial_coefficients
from pyamg.util.utils import scale_rows, get_block_diag, UnAmal
from pyamg.util.linalg import approximate_spectral_radius
from pyamg.krylov import gmres, cgne, cgnr, cg
from pyamg import amg_core
import scipy.lib.lapack as la

__docformat__ = "restructuredtext en"

__all__ = ['change_smoothers']

def unpack_arg(v):
    if isinstance(v,tuple):
        return v[0],v[1]
    else:
        return v,{}

def change_smoothers(ml, presmoother, postsmoother):
    '''
    Initialize pre- and post- smoothers throughout a multilevel_solver, with
    the option of having different smoothers at different levels

    For each level of the multilevel_solver 'ml' (except the coarsest level),
    initialize the .presmoother() and .postsmoother() methods used in the 
    multigrid cycle.

    Parameters
    ----------
    ml : {pyamg multilevel hierarchy}
        Data structure that stores the multigrid hierarchy.
    presmoother : {None, string, tuple, list}
        presmoother can be (1) the name of a supported smoother, 
        e.g. "gauss_seidel", (2) a tuple of the form ('method','opts') where 
        'method' is the name of a supported smoother and 'opts' a dict of
        keyword arguments to the smoother, or (3) a list of instances of options 1 or 2.
        See the Examples section for illustrations of the format. 

        If presmoother is a list, presmoother[i] determines the smoothing
        strategy for level i.  Else, presmoother defines the same strategy 
        for all levels.
        
        If len(presmoother) < len(ml.levels), then 
        presmoother[-1] is used for all remaining levels
        
        If len(presmoother) > len(ml.levels), then
        the remaining smoothing strategies are ignored
    
    postsmoother : {string, tuple, list}
        Defines postsmoother in identical fashion to presmoother

    Returns
    -------
    ml changed in place
    ml.level[i].presmoother  <===  presmoother[i]
    ml.level[i].postsmoother  <===  postsmoother[i]

    Notes
    -----
    - Parameter 'omega' of the Jacobi, Richardson, and jacobi_ne
      methods is scaled by the spectral radius of the matrix on 
      each level.  Therefore 'omega' should be in the interval (0,2).
    - By initializing the smoothers after the hierarchy has been setup, allows
      for "algebraically" directed relaxation, such as strength_based_schwarz,
      which uses only the strong connections of a degree-of-freedom to define
      overlapping regions
    - Available smoother methods::

        gauss_seidel
        block_gauss_seidel
        jacobi
        block_jacobi
        richardson
        sor
        chebyshev
        gauss_seidel_nr
        gauss_seidel_ne
        jacobi_ne
        cg
        gmres
        cgne
        cgnr
        schwarz
        strength_based_schwarz
        None

    Examples
    --------
    >>> from pyamg.gallery import poisson
    >>> from pyamg.aggregation import smoothed_aggregation_solver
    >>> from pyamg.relaxation.smoothing import change_smoothers
    >>> from pyamg.util.linalg import norm
    >>> from scipy import rand, array, mean
    >>> A = poisson((10,10), format='csr')
    >>> b = rand(A.shape[0],)
    >>> ml = smoothed_aggregation_solver(A, max_coarse=10)
    >>> #
    >>> ## Set all levels to use gauss_seidel's defaults
    >>> smoothers = 'gauss_seidel'
    >>> change_smoothers(ml, presmoother=smoothers, postsmoother=smoothers)
    >>> residuals=[]
    >>> x = ml.solve(b, tol=1e-8, residuals=residuals)
    >>> #
    >>> ## Set all levels to use three iterations of gauss_seidel's defaults
    >>> smoothers = ('gauss_seidel', {'iterations' : 3})
    >>> change_smoothers(ml, presmoother=smoothers, postsmoother=None)
    >>> residuals=[]
    >>> x = ml.solve(b, tol=1e-8, residuals=residuals)
    >>> #
    >>> ## Set level 0 to use gauss_seidel's defaults, and all  
    >>> ## subsequent levels to use 5 iterations of cgnr
    >>> smoothers = ['gauss_seidel', ('cgnr', {'maxiter' : 5})]
    >>> change_smoothers(ml, presmoother=smoothers, postsmoother=smoothers)
    >>> residuals=[]
    >>> x = ml.solve(b, tol=1e-8, residuals=residuals)
    '''

    # interpret arguments into list
    if isinstance(presmoother, str) or isinstance(presmoother, tuple) or (presmoother == None):
        presmoother = [presmoother]
    elif not isinstance(presmoother, list):
        raise ValueError,'Unrecognized presmoother'

    if isinstance(postsmoother, str) or isinstance(postsmoother, tuple) or (postsmoother == None):
        postsmoother = [postsmoother]
    elif not isinstance(postsmoother, list):
        raise ValueError,'Unrecognized postsmoother'

    # set ml.levels[i].presmoother = presmoother[i]
    i = 0
    for i in range( min(len(presmoother), len(ml.levels[:-1])) ):
        # unpack presmoother[i]
        fn,kwargs = unpack_arg(presmoother[i])
        # get function handle
        try:
            setup_presmoother = eval('setup_' + str(fn))
        except NameError, ne:
            raise NameError("invalid presmoother method: ", fn)
        ml.levels[i].presmoother = setup_presmoother(ml.levels[i], **kwargs)
    
    # Fill in remaining levels
    for j in range(i+1, len(ml.levels[:-1])):
        ml.levels[j].presmoother = setup_presmoother(ml.levels[j], **kwargs)


    # set ml.levels[i].postsmoother = postsmoother[i]
    i = 0
    for i in range( min(len(postsmoother), len(ml.levels[:-1])) ):
        # unpack postsmoother[i]
        fn,kwargs = unpack_arg(postsmoother[i])
        # get function handle
        try:
            setup_postsmoother = eval('setup_' + str(fn))
        except NameError, ne:
            raise NameError("invalid postsmoother method: ", fn)
        ml.levels[i].postsmoother = setup_postsmoother(ml.levels[i], **kwargs)
    
    # Fill in remaining levels
    for j in range(i+1, len(ml.levels[:-1])):
        ml.levels[j].postsmoother = setup_postsmoother(ml.levels[j], **kwargs)

def rho_D_inv_A(A):
    """
    Return the (approx.) spectral radius of D^-1 * A 
    
    Parameters
    ----------
    A : {sparse-matrix}

    Returns
    -------
    approximate spectral radius of diag(A)^{-1} A

    Examples
    --------
    >>> from pyamg.gallery import poisson
    >>> from pyamg.relaxation.smoothing import rho_D_inv_A
    >>> import numpy
    >>> A = numpy.array([[1,0,0],[0,2,0],[0,0,3]])
    >>> print rho_D_inv_A(A)
    1.0
    """
    
    if not hasattr(A, 'rho_D_inv'):
        D = A.diagonal()
        D_inv = 1.0 / D
        D_inv[D == 0] = 0
        D_inv_A = scale_rows(A, D_inv, copy=True)
        
        A.rho_D_inv = approximate_spectral_radius(D_inv_A)
    
    return A.rho_D_inv


def rho_block_D_inv_A(A, Dinv):
    """
    Return the (approx.) spectral radius of block D^-1 * A 
    
    Parameters
    ----------
    A : {sparse-matrix}
        size NxN
    Dinv : {array}
        Inverse of diagonal blocks of A
        size (N/blocksize, blocksize, blocksize)

    Returns
    -------
    approximate spectral radius of (Dinv A)

    Examples
    --------
    >>> from pyamg.gallery import poisson
    >>> from pyamg.relaxation.smoothing import rho_block_D_inv_A
    >>> from pyamg.util.utils import get_block_diag
    >>> import numpy
    >>> A = poisson((10,10), format='csr')
    >>> Dinv = get_block_diag(A, blocksize=4, inv_flag=True)
    >>> print rho_block_D_inv_A(A, Dinv)
    1.72254828808
    """

    if not hasattr(A, 'rho_block_D_inv'):
        from scipy.sparse.linalg import LinearOperator
        
        blocksize = Dinv.shape[1]
        if Dinv.shape[1] != Dinv.shape[2]:
            raise ValueError('Dinv has incorrect dimensions')
        elif Dinv.shape[0] != A.shape[0]/blocksize:
            raise ValueError('Dinv and A have incompatible dimensions')
        
        Dinv = scipy.sparse.bsr_matrix( (Dinv, \
                scipy.arange(Dinv.shape[0]), scipy.arange(Dinv.shape[0]+1)), shape=A.shape)
        
        # Don't explicitly form Dinv*A
        def matvec(x):
            return Dinv*(A*x)
        D_inv_A = LinearOperator(A.shape, matvec, dtype=A.dtype)
        
        A.rho_block_D_inv = approximate_spectral_radius(D_inv_A)

    return A.rho_block_D_inv

"""
    The following setup_smoother_name functions are helper functions for
    parsing user input and assigning each level the appropriate smoother for
    the above functions "change_smoothers".

    The standard interface is

    Parameters
    ----------
    lvl : {multilevel level}
        the level in the hierarchy for which to assign a smoother
    iterations : {int}
        how many smoother iterations
    optional_params : {}
        optional params specific for each method such as omega or sweep

    Returns
    -------
    Function pointer for the appropriate relaxation method for level=lvl

    Examples
    --------
    See change_smoothers above

"""
def setup_gauss_seidel(lvl, iterations=1, sweep='forward'):
    def smoother(A,x,b):
        relaxation.gauss_seidel(A, x, b, iterations=iterations, sweep=sweep)
    return smoother
        
def setup_jacobi(lvl, iterations=1, omega=1.0):
    omega = omega/rho_D_inv_A(lvl.A)
    def smoother(A,x,b):
        relaxation.jacobi(A, x, b, iterations=iterations, omega=omega)
    return smoother

def setup_schwarz(lvl, iterations=1, subdomain=None, subdomain_ptr=None, inv_subblock=None, inv_subblock_ptr=None):
    
    if not scipy.sparse.isspmatrix_csr(lvl.A):
        A = lvl.A.tocsr()
    else:
        A = lvl.A

    if subdomain is None or subdomain_ptr is None:
        subdomain_ptr = A.indptr.copy()
        subdomain = A.indices.copy()
        

    ##
    # Extract each subdomain's block from the matrix
    if inv_subblock is None or inv_subblock_ptr is None:
        inv_subblock_ptr = numpy.zeros(subdomain_ptr.shape, dtype=int)
        blocksize = (subdomain_ptr[1:] - subdomain_ptr[:-1])
        inv_subblock_ptr[1:] = numpy.cumsum(blocksize*blocksize)
        
        ##
        # Extract each block column from A
        inv_subblock = numpy.zeros((inv_subblock_ptr[-1],), dtype=A.dtype)
        amg_core.extract_subblocks(A.indptr, A.indices, A.data, inv_subblock, 
                          inv_subblock_ptr, subdomain, subdomain_ptr, 
                          subdomain_ptr.shape[0]-1, A.shape[0])
        
        ##
        # Invert each block column
        [my_pinv] = la.get_lapack_funcs(['gelss'], (numpy.ones((1,), dtype=A.dtype)) )
        for i in xrange(subdomain_ptr.shape[0]-1):
            m = blocksize[i]
            rhs = scipy.eye(m,m, dtype=A.dtype)
            [v,pseudo,s,rank,info] = \
                my_pinv(inv_subblock[inv_subblock_ptr[i]:inv_subblock_ptr[i+1]].reshape(m,m), rhs)
            inv_subblock[inv_subblock_ptr[i]:inv_subblock_ptr[i+1]] = numpy.ravel(pseudo)

    def smoother(A,x,b):
        relaxation.schwarz(A, x, b, iterations=iterations, subdomain=subdomain, \
                           subdomain_ptr=subdomain_ptr, inv_subblock=inv_subblock,\
                           inv_subblock_ptr=inv_subblock_ptr)
    return smoother


def setup_strength_based_schwarz(lvl, iterations=1):
    # Use the overlapping regions defined by the the strength of connection matrix C 
    # for the overlapping Schwarz method
    C = lvl.C.tocsr()
    subdomain_ptr = C.indptr.copy()
    subdomain = C.indices.copy()

    return setup_schwarz(lvl, iterations=iterations, subdomain=subdomain, subdomain_ptr=subdomain_ptr)


def setup_block_jacobi(lvl, iterations=1, omega=1.0, Dinv=None, blocksize=None):
    ##
    # Determine Blocksize
    if blocksize == None and Dinv == None:
        if scipy.sparse.isspmatrix_csr(lvl.A):
            blocksize = 1
        elif scipy.sparse.isspmatrix_bsr(lvl.A):
            blocksize = lvl.A.blocksize[0]
    elif blocksize == None:
        blocksize = Dinv.shape[1]
    
    if blocksize == 1:
        # Block Jacobi is equivalent to normal Jacobi
        return setup_jacobi(lvl, iterations=iterations, omega=omega)
    else:
        # Use Block Jacobi
        if Dinv == None:
            Dinv = get_block_diag(lvl.A, blocksize=blocksize, inv_flag=True)
        omega = omega/rho_block_D_inv_A(lvl.A, Dinv)
        def smoother(A,x,b):
            relaxation.block_jacobi(A, x, b, iterations=iterations, omega=omega, \
                                    Dinv=Dinv, blocksize=blocksize)
        return smoother

def setup_block_gauss_seidel(lvl, iterations=1, sweep='forward', Dinv=None, blocksize=None):
    ##
    # Determine Blocksize
    if blocksize == None and Dinv == None:
        if scipy.sparse.isspmatrix_csr(lvl.A):
            blocksize = 1
        elif scipy.sparse.isspmatrix_bsr(lvl.A):
            blocksize = lvl.A.blocksize[0]
    elif blocksize == None:
        blocksize = Dinv.shape[1]

    if blocksize == 1:
        # Block GS is equivalent to normal GS
        return setup_gauss_seidel(lvl, iterations=iterations, sweep=sweep)
    else:
        # Use Block GS       
        if Dinv == None:
            Dinv = get_block_diag(lvl.A, blocksize=blocksize, inv_flag=True)
        def smoother(A,x,b):
            relaxation.block_gauss_seidel(A, x, b, iterations=iterations, \
                               Dinv=Dinv, blocksize=blocksize, sweep=sweep)

        return smoother

def setup_richardson(lvl, iterations=1, omega=1.0):
    omega = omega/approximate_spectral_radius(lvl.A)
    def smoother(A,x,b):
        relaxation.polynomial(A, x, b, coeffients=[omega], iterations=iterations)
    return smoother

def setup_sor(lvl, omega=0.5, iterations=1, sweep='forward'):
    def smoother(A,x,b):
        relaxation.sor(A, x, b, omega=omega, iterations=iterations, sweep=sweep)
    return smoother

def setup_chebyshev(lvl, lower_bound=1.0/30.0, upper_bound=1.1, degree=3, iterations=1):
    rho = approximate_spectral_radius(lvl.A)
    a = rho * lower_bound
    b = rho * upper_bound
    coeffients = -chebyshev_polynomial_coefficients(a, b, degree)[:-1] # drop the constant coefficient
    def smoother(A,x,b):
        relaxation.polynomial(A, x, b, coeffients=coeffients, iterations=iterations)
    return smoother

def setup_jacobi_ne(lvl, iterations=1, omega=1.0):
    omega = omega/rho_D_inv_A(lvl.A)**2
    def smoother(A,x,b):
        relaxation.jacobi_ne(A, x, b, iterations=iterations, omega=omega)
    return smoother

def setup_gauss_seidel_ne(lvl, iterations=1, sweep='forward', omega=1.0):
    def smoother(A,x,b):
        relaxation.gauss_seidel_ne(A, x, b, iterations=iterations, sweep=sweep, omega=omega)
    return smoother

def setup_gauss_seidel_nr(lvl, iterations=1, sweep='forward', omega=1.0):
    def smoother(A,x,b):
        relaxation.gauss_seidel_nr(A, x, b, iterations=iterations, sweep=sweep, omega=omega)
    return smoother

def setup_gmres(lvl, tol=1e-12, maxiter=1, restrt=None, M=None, callback=None, residuals=None):
    def smoother(A,x,b):
        x[:] = (gmres(A, b, x0=x, tol=tol, maxiter=maxiter, restrt=restrt, M=M, callback=callback, residuals=residuals)[0]).reshape(x.shape)
    return smoother
            
def setup_cg(lvl, tol=1e-12, maxiter=1, M=None, callback=None, residuals=None):
    def smoother(A,x,b):
        x[:] = (cg(A, b, x0=x, tol=tol, maxiter=maxiter, M=M, callback=callback, residuals=residuals)[0]).reshape(x.shape)
    return smoother

def setup_cgne(lvl, tol=1e-12, maxiter=1, M=None, callback=None, residuals=None):
    def smoother(A,x,b):
        x[:] = (cgne(A, b, x0=x, tol=tol, maxiter=maxiter, M=M, callback=callback, residuals=residuals)[0]).reshape(x.shape)
    return smoother

def setup_cgnr(lvl, tol=1e-12, maxiter=1, M=None, callback=None, residuals=None):
    def smoother(A,x,b):
        x[:] = (cgnr(A, b, x0=x, tol=tol, maxiter=maxiter, M=M, callback=callback, residuals=residuals)[0]).reshape(x.shape)
    return smoother

def setup_None(lvl):
    def smoother(A,x,b):
        pass
    return smoother


