from numpy import array, zeros, ravel, inner, conjugate
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
    M : matrix-like
        n x n, inverted preconditioner, i.e. solve M A A.H x = b.
        For preconditioning with a mat-vec routine, set
        A.psolve = func, where func := M y
    callback : function
        callback( x ) is called each iteration, 
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
    
    # Check iteration numbers
    if maxiter == None:
        maxiter = len(x) + 5
    elif maxiter < 1:
        raise ValueError('Number of iterations must be positive')

    # Scale tol by normb
    normb = norm(b) 
    if normb != 0:
        tol = tol*normb

    # Prep for method
    r = b - ravel(A*x)
    normr = norm(r)

    if residuals is not None:
        residuals[:] = [normr]

    # Is initial guess sufficient?
    if normr < tol:
        return (postprocess(x), 0)
   
    rstar = r.copy()
    p     = r.copy()

    rrstarOld = inner(conjugate(rstar), r)

    iter = 0

    # Begin BiCGStab
    while True:
        Mp  = ravel(M*p)
        AMp = ravel(A*Mp)
        
        # alpha = (r_j, rstar) / (A*M*p_j, rstar)
        alpha = rrstarOld/inner(conjugate(rstar), AMp)
        
        # s_j = r_j - alpha*A*M*p_j
        s   = r - alpha*AMp
        Ms  = ravel(M*s)
        AMs = ravel(A*Ms)

        # omega = (A*M*s_j, s_j)/(A*M*s_j, A*M*s_j)
        omega = inner(conjugate(AMs), s)/inner(conjugate(AMs), AMs)

        # x_{j+1} = x_j +  alpha*M*p_j + omega*M*s_j
        x = x + alpha*Mp + omega*Ms

        # r_{j+1} = s_j - omega*A*M*s
        r = s - omega*AMs

        # beta_j = (r_{j+1}, rstar)/(r_j, rstar) * (alpha/omega)
        rrstarNew = inner(conjugate(rstar), r)
        beta      = (rrstarNew / rrstarOld) * (alpha / omega)
        rrstarOld = rrstarNew

        # p_{j+1} = r_{j+1} + beta*(p_j - omega*A*M*p)
        p = r + beta*(p - omega*AMp)

        iter += 1
        
        normr = norm(r)

        if residuals is not None:
            residuals.append(normr)

        if callback is not None:
            callback(x)

        if normr < tol:
            return (postprocess(x), 0)

        if iter == maxiter:
            return (postprocess(x), iter)

