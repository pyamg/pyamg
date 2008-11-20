from numpy import array, zeros, ravel, dot, conjugate
from scipy.sparse import csr_matrix, isspmatrix
from scipy.sparse.sputils import upcast
from scipy.sparse.linalg.isolve.utils import make_system
from scipy.sparse.linalg.interface import aslinearoperator
from scipy import ceil, asmatrix, rand
from warnings import warn
from pyamg.util.linalg import norm

__docformat__ = "restructuredtext en"

__all__ = ['bicgstab']


def bicgstab(A, b, x0=None, tol=1e-5, maxiter=None, xtype=None, M=None, callback=None, residuals=None):
    '''
    Biconjugate Gradient Aglorithm with Stabilization applied to A x = b
    Left preconditioning is supported

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
        n x n, inverted preconditioner, i.e. solve M A A.H x = b.
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
    >>>(x,flag) = bicgstab(A,b)
    >>>print pyamg.util.linalg.norm(b - A*x)

    References
    ----------
    Yousef Saad, "Iterative Methods for Sparse Linear Systems, 
    Second Edition", SIAM, pp. 231-234, 2003

    '''
    
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
    if maxiter == None:
        maxiter = dimen + 5
    elif maxiter < 1:
        raise ValueError('Number of iterations must be positive')

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
    normr = norm(r)
    if keep_r:
        residuals.append(normr)

    # Is initial guess sufficient?
    if normr <= tol:
        if callback != None:    
            callback(r)
        
        return (postprocess(x), 0)
   
    rstar = r.copy()
    p = r.copy()
    rrstarOld = dot(conjugate(rstar), r)

    # Begin BiCGStab
    for iter in range(maxiter):
        
        Mp = ravel(M*p)
        AMp = ravel(A*Mp)
        
        # alpha = (r_j, rstar) / (A*M*p_j, rstar)
        alpha = rrstarOld/dot(conjugate(rstar), AMp)
        
        # s_j = r_j - alpha*A*M*p_j
        s = r - alpha*AMp
        Ms = ravel(M*s)
        AMs = ravel(A*Ms)

        # omega = (A*M*s_j, s_j)/(A*M*s_j, A*M*s_j)
        omega = dot(conjugate(AMs), s)/dot(conjugate(AMs), AMs)

        # x_{j+1} = x_j +  alpha*M*p_j + omega*M*s_j
        x = x + alpha*Mp + omega*Ms

        # r_{j+1} = s_j - omega*A*M*s
        r = s - omega*AMs

        # beta_j = (r_{j+1}, rstar)/(r_j, rstar) * (alpha/omega)
        rrstarNew = dot(conjugate(rstar), r)
        beta = (rrstarNew/rrstarOld)*(alpha/omega)
        rrstarOld = rrstarNew

        # p_{j+1} = r_{j+1} + beta*(p_j - omega*A*M*p)
        p = r + beta*(p - omega*AMp)

        # Allow user access to residual
        if callback != None:
            callback( r )
        
        # test for convergence
        normr = norm(r)
        if keep_r:
            residuals.append(normr)
        if normr < tol:
            return (postprocess(x),0)

    # end loop

    return (postprocess(x), iter+1)

