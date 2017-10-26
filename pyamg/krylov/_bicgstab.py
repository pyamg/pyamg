import numpy as np
from scipy.sparse.linalg.isolve.utils import make_system
from pyamg.util.linalg import norm


__all__ = ['bicgstab']


def bicgstab(A, b, x0=None, tol=1e-5, maxiter=None, xtype=None, M=None,
             callback=None, residuals=None):
    '''Biconjugate Gradient Algorithm with Stabilization

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
        relative convergence tolerance, i.e. tol is scaled by ||r_0||_2
    maxiter : int
        maximum number of allowed iterations
    xtype : type
        dtype for the solution, default is automatic type detection
    M : {array, matrix, sparse matrix, LinearOperator}
        n x n, inverted preconditioner, i.e. solve M A A.H x = M b.
    callback : function
        User-supplied function is called after each iteration as
        callback(xk), where xk is the current solution vector
    residuals : list
        residuals has the residual norm history,
        including the initial residual, appended to it

    Returns
    -------
    (xNew, info)
    xNew : an updated guess to the solution of Ax = b
    info : halting status of bicgstab

            ==  ======================================
            0   successful exit
            >0  convergence to tolerance not achieved,
                return iteration count instead.
            <0  numerical breakdown, or illegal input
            ==  ======================================

    Notes
    -----
    The LinearOperator class is in scipy.sparse.linalg.interface.
    Use this class if you prefer to define A or M as a mat-vec routine
    as opposed to explicitly constructing the matrix.  A.psolve(..) is
    still supported as a legacy.

    Examples
    --------
    >>> from pyamg.krylov.bicgstab import bicgstab
    >>> from pyamg.util.linalg import norm
    >>> import numpy as np
    >>> from pyamg.gallery import poisson
    >>> A = poisson((10,10))
    >>> b = np.ones((A.shape[0],))
    >>> (x,flag) = bicgstab(A,b, maxiter=2, tol=1e-8)
    >>> print norm(b - A*x)
    4.68163045309

    References
    ----------
    .. [1] Yousef Saad, "Iterative Methods for Sparse Linear Systems,
       Second Edition", SIAM, pp. 231-234, 2003
       http://www-users.cs.umn.edu/~saad/books.html

    '''

    # Convert inputs to linear system, with error checking
    A, M, x, b, postprocess = make_system(A, M, x0, b)

    # Ensure that warnings are always reissued from this function
    import warnings
    warnings.filterwarnings('always', module='pyamg\.krylov\._bicgstab')

    # Check iteration numbers
    if maxiter is None:
        maxiter = len(x) + 5
    elif maxiter < 1:
        raise ValueError('Number of iterations must be positive')

    # Prep for method
    r = b - A*x
    normr = norm(r)

    if residuals is not None:
        residuals[:] = [normr]

    # Check initial guess ( scaling by b, if b != 0,
    #   must account for case when norm(b) is very small)
    normb = norm(b)
    if normb == 0.0:
        normb = 1.0
    if normr < tol*normb:
        return (postprocess(x), 0)

    # Scale tol by ||r_0||_2
    if normr != 0.0:
        tol = tol*normr

    # Is this a one dimensional matrix?
    if A.shape[0] == 1:
        entry = np.ravel(A*np.array([1.0], dtype=xtype))
        return (postprocess(b/entry), 0)

    rstar = r.copy()
    p = r.copy()

    rrstarOld = np.inner(rstar.conjugate(), r)

    iter = 0

    # Begin BiCGStab
    while True:
        Mp = M*p
        AMp = A*Mp

        # alpha = (r_j, rstar) / (A*M*p_j, rstar)
        alpha = rrstarOld/np.inner(rstar.conjugate(), AMp)

        # s_j = r_j - alpha*A*M*p_j
        s = r - alpha*AMp
        Ms = M*s
        AMs = A*Ms

        # omega = (A*M*s_j, s_j)/(A*M*s_j, A*M*s_j)
        omega = np.inner(AMs.conjugate(), s)/np.inner(AMs.conjugate(), AMs)

        # x_{j+1} = x_j +  alpha*M*p_j + omega*M*s_j
        x = x + alpha*Mp + omega*Ms

        # r_{j+1} = s_j - omega*A*M*s
        r = s - omega*AMs

        # beta_j = (r_{j+1}, rstar)/(r_j, rstar) * (alpha/omega)
        rrstarNew = np.inner(rstar.conjugate(), r)
        beta = (rrstarNew / rrstarOld) * (alpha / omega)
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

# if __name__ == '__main__':
#    # from numpy import diag
#    # A = random((4,4))
#    # A = A*A.transpose() + diag([10,10,10,10])
#    # b = random((4,1))
#    # x0 = random((4,1))
#    # %timeit -n 15 (x,flag) = bicgstab(A,b,x0,tol=1e-8,maxiter=100)
#    from pyamg.gallery import stencil_grid
#    from numpy.random import random
#    A = stencil_grid([[0,-1,0],[-1,4,-1],[0,-1,0]],(100,100),
#                     dtype=float,format='csr')
#    b = random((A.shape[0],))
#    x0 = random((A.shape[0],))
#
#    import time
#    from scipy.sparse.linalg.isolve import bicgstab as ibicgstab
#
#    print '\n\nTesting BiCGStab with %d x %d 2D Laplace Matrix' % \
#           (A.shape[0],A.shape[0])
#    t1=time.time()
#    (x,flag) = bicgstab(A,b,x0,tol=1e-8,maxiter=100)
#    t2=time.time()
#    print '%s took %0.3f ms' % ('bicgstab', (t2-t1)*1000.0)
#    print 'norm = %g'%(norm(b - A*x))
#    print 'info flag = %d'%(flag)
#
#    t1=time.time()
#    (y,flag) = ibicgstab(A,b,x0,tol=1e-8,maxiter=100)
#    t2=time.time()
#    print '\n%s took %0.3f ms' % ('linalg bicgstab', (t2-t1)*1000.0)
#    print 'norm = %g'%(norm(b - A*y))
#    print 'info flag = %d'%(flag)
