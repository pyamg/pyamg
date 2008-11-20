from numpy import dot, conjugate, ravel, array, int, ceil
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
    >>>(x,flag) = cg(A,b)
    >>>print pyamg.util.linalg.norm(b - A*x)

    References
    ----------
    Yousef Saad, "Iterative Methods for Sparse Linear Systems, 
    Second Edition", SIAM, pp. 262-67, 2003

    '''
    A,M,x,b,postprocess = make_system(A,M,x0,b,xtype=None)

    # We assume henceforth that shape=(n,) for all arrays
    xtype = upcast(A.dtype, x.dtype, b.dtype, M.dtype)
    b = ravel(array(b,xtype))
    x = ravel(array(x,xtype))
    
    n = len(b)
    # Determine maxiter
    if maxiter is None:
        maxiter = int(ceil(1.3*n)) + 2
    elif maxiter < 1:
        raise ValueError('Number of iterations must be positive')
    
    # Should norm(r) be kept
    if residuals == []:
        keep_r = True
    else:
        keep_r = False
    
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

    # setup method
    doneiterating = False
    iter = 0
    flag = 1

    r = b - ravel(A*x)

    normr0 = norm(r)
    if keep_r:
        residuals.append(normr0)

    if normr0 < tol:
        doneiterating = True

    z = ravel(M*r)
    rz = dot(conjugate(ravel(r)), ravel(z))
    p = z.copy()

    while not doneiterating:
        Ap = ravel(A*p)
        alpha = rz/dot(conjugate(ravel(Ap)), ravel(p))

        x += alpha * p

        r -= alpha * Ap
        z = ravel(M*r)

        rz_new = dot(conjugate(ravel(r)), ravel(z))
        beta = rz_new/rz
        rz = rz_new
        
        # Bizzare Behavior
        z = z + beta*p
        #z += beta * p
        p = z.copy()

        iter += 1
        
        normr = norm(r)
        if keep_r:
            residuals.append(normr)
        
        if callback != None:
            callback(r)

        if normr < tol:
            doneiterating = True
            flag = 0

        if iter > (maxiter-1):
            doneiterating = True
    
    if flag == 0:
        return (postprocess(x), flag)
    else:    
        return (postprocess(x), iter)

if __name__ == '__main__':
    # from numpy import diag
    # A = random((4,4))
    # A = A*A.transpose() + diag([10,10,10,10])
    # b = random((4,1))
    # x0 = random((4,1))

    from pyamg.gallery import stencil_grid
    from numpy.random import random
    A = stencil_grid([[0,-1,0],[-1,4,-1],[0,-1,0]],(10,10),dtype=float,format='csr')
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

    
