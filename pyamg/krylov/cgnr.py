from numpy import array, zeros, ravel, dot, conjugate
from scipy.sparse import csr_matrix, isspmatrix
from scipy.sparse.sputils import upcast
from scipy.sparse.linalg.isolve.utils import make_system
from scipy.sparse.linalg.interface import aslinearoperator
from scipy import ceil, asmatrix
from warnings import warn
from pyamg.util.linalg import norm

__docformat__ = "restructuredtext en"

__all__ = ['cgnr']

def cgnr(A, b, x0=None, tol=1e-5, maxiter=None, xtype=None, M=None, callback=None, residuals=None):
    '''
    Conjugate Gradient, Normal Residual algorithm
    Applies CG to the normal equations, A.H A x = b
    Left preconditioning is supported
    Note that if A is not well-conditioned, this algorithm is unadvisable

    Parameters
    ----------
    A : array, matrix or sparse matrix
        n x n, linear system to solve
    b : array
        n x 1, right hand side
    x0 : array
        n x 1, initial guess
        default is a vector of zeros
    tol : float
        convergence tolerance
    maxiter : int
        maximum number of allowed iterations
        default is A.shape[0]
    xtype : type
        dtype for the solution
    M : matrix-like
        n x n, inverted preconditioner, i.e. solve M A.H A x = b.
        For preconditioning with a mat-vec routine, set
        A.psolve = func, where func := M y
    callback : function
        callback( r ) is called each iteration, 
        where r = b - A*x 
    residuals : {None, empty-list}
        If empty-list, residuals holds the residual norm history,
        including the initial residual, upon completion
     
    Returns
    -------    
    (xNew, info)
    xNew -- an updated guess to the solution of Ax = b
    info -- halting status of gmres
            0  : successful exit
            >0 : convergence to tolerance not achieved,
                 return iteration count instead.  
            <0 : numerical breakdown, or illegal input


    Notes
    -----
    
    Examples
    --------
    >>>from pyamg.krylov import *
    >>>from scipy import rand
    >>>import pyamg
    >>>A = pyamg.poisson((50,50))
    >>>b = rand(A.shape[0],)
    >>>(x,flag) = cgnr(A,b)
    >>>print pyamg.util.linalg.norm(b - A*x)

    References
    ----------
    Yousef Saad, "Iterative Methods for Sparse Linear Systems, 
    Second Edition", SIAM, pp. 276-7, 2003

    '''

    # Store the conjugate transpose explicitly as it will be used much later on
    if isspmatrix(A):
        AH = A.tocsr().conjugate().transpose()
    else:
        AH = aslinearoperator(asmatrix(A).H)
    
    # Convert inputs to linear system, with error checking  
    A,M,x,b,postprocess = make_system(A,M,x0,b,xtype)
    dimen = A.shape[0]
    
    # Choose type
    xtype = upcast(A.dtype, x.dtype, b.dtype, M.dtype)

    # We assume henceforth that shape=(n,) for all arrays
    b = ravel(array(b,xtype))
    x = ravel(array(x,xtype))
    
    # Should norm(r) be kept
    if residuals == []:
        keep_r = True
    else:
        keep_r = False

    # Check iteration numbers
    # CGNE suffers from loss of orthogonality quite easily, so we arbitarily let the method go up to 130% over the
    # theoretically necessary limit of maxiter=dimen
    if maxiter == None:
        maxiter = int(ceil(1.3*dimen)) + 2
    elif maxiter < 1:
        raise ValueError('Number of iterations must be positive')
    elif maxiter > (1.3*dimen):
        warn('maximimum allowed inner iterations (maxiter) are the 130% times the number of degress of freedom')
        maxiter = int(ceil(1.3*dimen)) + 2

    # Scale tol by normb
    normb = norm(b) 
    if normb == 0:
        pass
    #    if callback != None:
    #        callback(0.0)
    #
    #    return (postprocess(zeros((dimen,), dtype=xtype)),0)
    else:
        tol = tol*normb

    # Prep for method
    r = b - ravel(A*x)
    rhat = AH*r
    normr = norm(r)
    if keep_r:
        residuals.append(normr)

    # Is initial guess sufficient?
    if normr <= tol:
        if callback != None:    
            callback(r)
        
        return (postprocess(x), 0)
   
    # Begin CGNR

    # Apply preconditioner and calculate initial search direction
    z = ravel(M*rhat)
    p = z.copy()
    old_zr = dot(conjugate(z), rhat)

    for iter in range(maxiter):

        # w_j = A p_j
        w = ravel(A*p)

        # alpha = (z_j, rhat_j) / (w_j, w_j)
        alpha = old_zr/dot(conjugate(w), w)
        
        # x_{j+1} = x_j + alpha*p_j
        x = x + alpha*p

        # r_{j+1} = r_j - alpha*w_j
        r = r - alpha*w

        # rhat_{j+1} = A.H*r_{j+1}
        rhat = ravel(AH*r)

        # z_{j+1} = M*r_{j+1}
        z = ravel(M*rhat)

        # beta = (z_{j+1}, rhat_{j+1}) / (z_j, rhat_j)
        new_zr = dot(conjugate(z), rhat)
        beta = new_zr / old_zr
        old_zr = new_zr

        # p_{j+1} = A.H*z_{j+1} + beta*p_j
        p = z + beta*p

        # Allow user access to residual
        if callback != None:
            callback( r )
        
        # test for convergence
        normr = norm(r)
        if keep_r:
            residuals.append(normr)
        if normr < tol:
            return (postprocess(x), 0)

    # end loop

    return (postprocess(x), iter+1)

