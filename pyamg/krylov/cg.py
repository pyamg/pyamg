from numpy import inner, conjugate, ravel, asarray, int, ceil
from scipy.sparse.linalg.isolve.utils import make_system
from scipy.sparse.sputils import upcast
from pyamg.util.linalg import norm
from warnings import warn

__docformat__ = "restructuredtext en"

__all__=['cg']

def cg(A, b, x0=None, tol=1e-5, maxiter=None, xtype=None, M=None, callback=None, residuals=None):
    '''
    Conjugate Gradient on A x = b
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
        default is A.shape[0]/10
    M : matrix-like
        n x n, inverted preconditioner, i.e. solve M A A.H x = b.
        For preconditioning with a mat-vec routine, set
        A.psolve = func, where func := M y
    callback : function
        callback(x) is after each iteration, 
    residuals : {None, list}
        If not None, residuals holds the residual norm history, including 
        the initial residual, upon completion.
     
    Returns
    -------    
    (xNew, info)
    xNew -- an updated guess to the solution of Ax = b
    info -- halting status of cg
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
    >>>(x,flag) = cg(A,b)
    >>>print pyamg.util.linalg.norm(b - A*x)

    References
    ----------
    Yousef Saad, "Iterative Methods for Sparse Linear Systems, 
    Second Edition", SIAM, pp. 262-67, 2003

    '''
    A,M,x,b,postprocess = make_system(A,M,x0,b,xtype=None)
    x = ravel(x)
    b = ravel(b)

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
    r  = b - ravel(A*x)
    z  = ravel(M*r)
    p  = z.copy()
    rz = inner(conjugate(r), z)
    
    normr = norm(r)

    if residuals is not None:
        residuals[:] = [normr] #initial residual 

    if normr < tol:
        return (postprocess(x), 0)

    iter = 0

    while True:
        Ap = ravel(A*p)

        rz_old = rz
        
        alpha = rz/inner(conjugate(Ap), p)  # 3  (step # in Saad's pseudocode)
        x    += alpha * p                   # 4
        r    -= alpha * Ap                  # 5
        z     = ravel(M*r)                  # 6
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

if __name__ == '__main__':
    # from numpy import diag
    # A = random((4,4))
    # A = A*A.transpose() + diag([10,10,10,10])
    # b = random((4,1))
    # x0 = random((4,1))

    from pyamg.gallery import stencil_grid
    from numpy.random import random
    A = stencil_grid([[0,-1,0],[-1,4,-1],[0,-1,0]],(100,100),dtype=float,format='csr')
    b = random((A.shape[0],))
    x0 = random((A.shape[0],))

    import time
    from scipy.sparse.linalg.isolve import cg as icg

    print '\n\nTesting CG with %d x %d 2D Laplace Matrix'%(A.shape[0],A.shape[0])
    t1=time.time()
    (x,flag) = cg(A,b,x0,tol=1e-8,maxiter=100)
    t2=time.time()
    print '%s took %0.3f ms' % ('cg', (t2-t1)*1000.0)
    print 'norm = %g'%(norm(b - A*x))
    print 'info flag = %d'%(flag)

    t1=time.time()
    (y,flag) = icg(A,b,x0,tol=1e-8,maxiter=100)
    t2=time.time()
    print '\n%s took %0.3f ms' % ('linalg cg', (t2-t1)*1000.0)
    print 'norm = %g'%(norm(b - A*y))
    print 'info flag = %d'%(flag)

    
