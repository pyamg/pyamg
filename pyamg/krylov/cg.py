from numpy import inner, conjugate, asarray 
from scipy.sparse.linalg.isolve.utils import make_system
from scipy.sparse.sputils import upcast
from pyamg.util.linalg import norm
from warnings import warn

__docformat__ = "restructuredtext en"

__all__ = ['cg']

def cg(A, b, x0=None, tol=1e-5, maxiter=None, xtype=None, M=None, callback=None, residuals=None):
    '''Conjugate Gradient algorithm
    
    Solves the linear system Ax = b. Left preconditioning is supported.

    Parameters
    ----------
    A : {array, matrix, sparse matrix, LinearOperator}
        n x n, linear system to solve
    b : {array, matrix}
        right hand side, shape is (n,) or (n,1)
    x0 : {array, matrix}
        initial guess, default is a vector of zeros
    tol : float
        relative convergence tolerance, i.e. tol is scaled by ||b||
    maxiter : int
        maximum number of allowed iterations
    xtype : type
        dtype for the solution, default is automatic type detection
    M : {array, matrix, sparse matrix, LinearOperator}
        n x n, inverted preconditioner, i.e. solve M A A.H x = b.
    callback : function
        User-supplied funtion is called after each iteration as
        callback(xk), where xk is the current solution vector
    residuals : list
        residuals has the residual norm history,
        including the initial residual, appended to it
     
    Returns
    -------    
    (xNew, info)
    xNew : an updated guess to the solution of Ax = b
    info : halting status of cg

            ==  ======================================= 
            0   successful exit
            >0  convergence to tolerance not achieved,
                return iteration count instead.  
            <0  numerical breakdown, or illegal input
            ==  ======================================= 

    Notes
    -----
    The LinearOperator class is in scipy.sparse.linalg.interface.
    Use this class if you prefer to define A or M as a mat-vec routine
    as opposed to explicitly constructing the matrix.  A.psolve(..) is
    still supported as a legacy.

    Examples
    --------
    >>> from pyamg.krylov.cg import cg
    >>> from pyamg.util.linalg import norm
    >>> import numpy 
    >>> from pyamg.gallery import poisson
    >>> A = poisson((10,10))
    >>> b = numpy.ones((A.shape[0],))
    >>> (x,flag) = cg(A,b, maxiter=2, tol=1e-8)
    >>> print norm(b - A*x)
    10.9370700187

    References
    ----------
    .. [1] Yousef Saad, "Iterative Methods for Sparse Linear Systems, 
       Second Edition", SIAM, pp. 262-67, 2003
       http://www-users.cs.umn.edu/~saad/books.html

    '''
    A,M,x,b,postprocess = make_system(A,M,x0,b,xtype=None)

    n = len(b)
    # Determine maxiter
    if maxiter is None:
        maxiter = int(1.3*len(b)) + 2
    elif maxiter < 1:
        raise ValueError('Number of iterations must be positive')
    
    # Scale tol by normb
    normb = norm(b) 
    if normb != 0:
        tol = tol*normb

    # setup method
    r  = b - A*x
    z  = M*r
    p  = z.copy()
    rz = inner(conjugate(r), z)
    
    normr = norm(r)

    if residuals is not None:
        residuals[:] = [normr] #initial residual 

    if normr < tol:
        return (postprocess(x), 0)

    iter = 0

    while True:
        Ap = A*p

        rz_old = rz
        
        alpha = rz/inner(conjugate(Ap), p)  # 3  (step # in Saad's pseudocode)
        x    += alpha * p                   # 4
        r    -= alpha * Ap                  # 5
        z     = M*r                         # 6
        rz    = inner(conjugate(r), z)          
        beta  = rz/rz_old                   # 7
        p    *= beta                        # 8
        p    += z

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

#if __name__ == '__main__':
#    # from numpy import diag
#    # A = random((4,4))
#    # A = A*A.transpose() + diag([10,10,10,10])
#    # b = random((4,1))
#    # x0 = random((4,1))
#
#    from pyamg.gallery import stencil_grid
#    from numpy.random import random
#    A = stencil_grid([[0,-1,0],[-1,4,-1],[0,-1,0]],(100,100),dtype=float,format='csr')
#    b = random((A.shape[0],))
#    x0 = random((A.shape[0],))
#
#    import time
#    from scipy.sparse.linalg.isolve import cg as icg
#
#    print '\n\nTesting CG with %d x %d 2D Laplace Matrix'%(A.shape[0],A.shape[0])
#    t1=time.time()
#    (x,flag) = cg(A,b,x0,tol=1e-8,maxiter=100)
#    t2=time.time()
#    print '%s took %0.3f ms' % ('cg', (t2-t1)*1000.0)
#    print 'norm = %g'%(norm(b - A*x))
#    print 'info flag = %d'%(flag)
#
#    t1=time.time()
#    (y,flag) = icg(A,b,x0,tol=1e-8,maxiter=100)
#    t2=time.time()
#    print '\n%s took %0.3f ms' % ('linalg cg', (t2-t1)*1000.0)
#    print 'norm = %g'%(norm(b - A*y))
#    print 'info flag = %d'%(flag)
#
#    
