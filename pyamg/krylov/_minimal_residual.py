import numpy as np
from scipy.sparse.linalg.isolve.utils import make_system
from pyamg.util.linalg import norm
from warnings import warn


__all__ = ['minimal_residual']


def minimal_residual(A, b, x0=None, tol=1e-5, maxiter=None, xtype=None, M=None,
                     callback=None, residuals=None):
    '''Minimal residual (MR) algorithm

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
        relative convergence tolerance, i.e. tol is scaled by the
        preconditioner norm of r_0, or ||r_0||_M.
    maxiter : int
        maximum number of allowed iterations
    xtype : type
        dtype for the solution, default is automatic type detection
    M : {array, matrix, sparse matrix, LinearOperator}
        n x n, inverted preconditioner, i.e. solve M A x = M b.
    callback : function
        User-supplied function is called after each iteration as
        callback(xk), where xk is the current solution vector
    residuals : list
        residuals contains the residual norm history,
        including the initial residual.  The preconditioner norm
        is used, instead of the Euclidean norm.

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

    The residual in the preconditioner norm is both used for halting and
    returned in the residuals list.

    Examples
    --------
    >>> from pyamg.krylov import minimal_residual
    >>> from pyamg.util.linalg import norm
    >>> import numpy as np
    >>> from pyamg.gallery import poisson
    >>> A = poisson((10,10))
    >>> b = np.ones((A.shape[0],))
    >>> (x,flag) = minimal_residual(A,b, maxiter=2, tol=1e-8)
    >>> print norm(b - A*x)
    7.26369350856

    References
    ----------
    .. [1] Yousef Saad, "Iterative Methods for Sparse Linear Systems,
       Second Edition", SIAM, pp. 137--142, 2003
       http://www-users.cs.umn.edu/~saad/books.html

    '''
    A, M, x, b, postprocess = make_system(A, M, x0, b)

    # Ensure that warnings are always reissued from this function
    import warnings
    warnings.filterwarnings('always',
                            module='pyamg\.krylov\._minimal_residual')

    # determine maxiter
    if maxiter is None:
        maxiter = int(len(b))
    elif maxiter < 1:
        raise ValueError('Number of iterations must be positive')

    # setup method
    r = M*(b - A*x)
    normr = norm(r)

    # store initial residual
    if residuals is not None:
        residuals[:] = [normr]

    # Check initial guess ( scaling by b, if b != 0,
    #   must account for case when norm(b) is very small)
    normb = norm(b)
    if normb == 0.0:
        normb = 1.0
    if normr < tol*normb:
        return (postprocess(x), 0)

    # Scale tol by ||r_0||_M
    if normr != 0.0:
        tol = tol*normr

    # How often should r be recomputed
    recompute_r = 50

    iter = 0

    while True:
        iter = iter+1

        p = M*(A*r)

        rMAr = np.inner(p.conjugate(), r)  # check curvature of M^-1 A
        if rMAr < 0.0:
            warn("\nIndefinite matrix detected in minimal residual,\
                  aborting\n")
            return (postprocess(x), -1)

        alpha = rMAr / np.inner(p.conjugate(), p)
        x = x + alpha*r

        if np.mod(iter, recompute_r) and iter > 0:
            r = M*(b - A*x)
        else:
            r = r - alpha*p

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
#
#    from pyamg.gallery import stencil_grid
#    from pyamg import smoothed_aggregation_solver
#    from numpy.random import random
#    from numpy import zeros_like, dot
#    A = stencil_grid([[0,-1,0],[-1,4,-1],[0,-1,0]],(100,100),dtype=float,
#                     format='csr')
#    x0 = random((A.shape[0],))
#    b = zeros_like(x0)
#
#    # This function should always decrease (assuming zero RHS)
#    fvals = []
#    def callback(x):
#        fvals.append( sqrt(dot( ravel(x), ravel(A*x.reshape(-1,1)) )) )
#
#    print '\n\nTesting minimal residual with %d x %d 2D Laplace Matrix' %\
#          (A.shape[0],A.shape[0])
#    resvec = []
#    sa = smoothed_aggregation_solver(A)
#    #(x,flag) = minimal_residual(A,b,x0,tol=1e-8,maxiter=20,residuals=resvec,
#    M=sa.aspreconditioner())
#    (x,flag) = minimal_residual(A,b,x0,tol=1e-8,maxiter=20,residuals=resvec,
#    callback=callback)
#    print 'Funcation values:  ' + str(fvals)
#    print 'initial norm = %g'%(norm(b - A*x0))
#    print 'norm = %g'%(norm(b - A*x))
#    print 'info flag = %d'%(flag)
