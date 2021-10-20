import warnings
from warnings import warn
import numpy as np
from scipy.sparse.linalg.isolve.utils import make_system
from pyamg.util.linalg import norm


__all__ = ['minimal_residual']


def minimal_residual(A, b, x0=None, tol=1e-5, normA=None,
                     maxiter=None, M=None,
                     callback=None, residuals=None):
    """Minimal residual (MR) algorithm.

    Solves the linear system Ax = b. Left preconditioning is supported.

    Parameters
    ----------
    A : array, matrix, sparse matrix, LinearOperator
        n x n, linear system to solve
    b : array, matrix
        right hand side, shape is (n,) or (n,1)
    x0 : array, matrix
        initial guess, default is a vector of zeros
    tol : float
        stopping criteria (see normA)
        ||r_k|| < tol * ||b||, 2-norms
    normA : float
        if provided, then the stopping criteria becomes
        ||r_k|| < tol * (normA * ||x_k|| + ||b||), 2-norms
    maxiter : int
        maximum number of iterations allowed
    M : array, matrix, sparse matrix, LinearOperator
        n x n, inverted preconditioner, i.e. solve M A x = M b.
    callback : function
        User-supplied function is called after each iteration as
        callback(xk), where xk is the current solution vector
    residuals : list
        residual history in the 2-norm, including the initial residual

    Returns
    -------
    (xk, info)
    xk : an updated guess after k iterations to the solution of Ax = b
    info : halting status

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
    as opposed to explicitly constructing the matrix.

    Examples
    --------
    >>> from pyamg.krylov import minimal_residual
    >>> from pyamg.util.linalg import norm
    >>> import numpy as np
    >>> from pyamg.gallery import poisson
    >>> A = poisson((10,10))
    >>> b = np.ones((A.shape[0],))
    >>> (x,flag) = minimal_residual(A,b, maxiter=2, tol=1e-8)
    >>> print norm(b - A @ x)
    7.26369350856

    References
    ----------
    .. [1] Yousef Saad, "Iterative Methods for Sparse Linear Systems,
       Second Edition", SIAM, pp. 137--142, 2003
       http://www-users.cs.umn.edu/~saad/books.html

    """
    A, M, x, b, postprocess = make_system(A, M, x0, b)

    # Ensure that warnings are always reissued from this function
    warnings.filterwarnings('always', module='pyamg.krylov._minimal_residual')

    # determine maxiter
    if maxiter is None:
        maxiter = int(1.3*len(b)) + 2
    elif maxiter < 1:
        raise ValueError('Number of iterations must be positive')

    # setup method
    r = b - A @ x
    normr = norm(r)
    r = M @ r

    # store initial residual
    if residuals is not None:
        residuals[:] = [normr]

    # Check initial guess if b != 0,
    # must account for case when norm(b) is very small)
    normb = norm(b)
    if normb == 0.0 and normA:
        normb = 1.0
    if normA is not None:
        rtol = tol * (normA * np.linalg.norm(x) + normb)
    else:
        rtol = tol * normb
    if normr < rtol:
        return (postprocess(x), 0)

    # How often should r be recomputed
    recompute_r = 50

    it = 0

    while True:
        p = M @ (A @ r)

        rMAr = np.inner(p.conjugate(), r)  # check curvature of M^-1 A
        if rMAr < 0.0:
            warn("\nIndefinite matrix detected in minimal residual,\
                  aborting\n")
            return (postprocess(x), -1)

        alpha = rMAr / np.inner(p.conjugate(), p)
        x = x + alpha * r

        it += 1
        if np.mod(it, recompute_r) and it > 0:
            r = M @ (b - A @ x)
        else:
            r = r - alpha * p

        normr = norm(b - A @ x)
        if residuals is not None:
            residuals.append(normr)

        if callback is not None:
            callback(x)

        if normA is not None:
            rtol = tol * (normA * np.linalg.norm(x) + normb)
        else:
            rtol = tol * normb

        if normr < tol:
            return (postprocess(x), 0)

        if it == maxiter:
            return (postprocess(x), it)


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
