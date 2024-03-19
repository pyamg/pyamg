"""Short Recurrence Enlarged Conjugate Gradient algorithm."""
import warnings

import numpy as np

from ..util import make_system
from ..util.linalg import norm, bcgs, cgs, split_residual


def srecg_bcgs(A, b, x0=None, t=1, tol=1e-5, maxiter=None, M=None,
               callback=None, residuals=None):
    """Short Recurrence Enlarged Conjugate Gradient algorithm.

    Solves the linear system Ax = b. Left preconditioning is supported.

    Parameters
    ----------
    A : {array, matrix, sparse matrix, LinearOperator}
        n x n, linear system to solve
    b : {array, matrix}
        right hand side, shape is (n,) or (n,1)
    t : int
        number of partitions in which to split x0
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
    info : halting status of srecg

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
    >>> from pyamg.krylov.srecg import srecg
    >>> from pyamg.util.linalg import norm
    >>> import numpy as np
    >>> from pyamg.gallery import poisson
    >>> A = poisson((10,10))
    >>> b = np.ones((A.shape[0],))
    >>> (x,flag) = srecg(A,b, maxiter=2, tol=1e-8)
    >>> print norm(b - A*x)
    10.9370700187

    References
    ----------
    .. [1] Grigori, Laura, Sophie Moufawad, and Frederic Nataf.
       "Enlarged Krylov Subspace Conjugate Gradient Methods for Reducing
       Communication", SIAM Journal on Matrix Analysis and Applications 37(2),
       pp. 744-773, 2016.

    """
    A, M, x, b, postprocess = make_system(A, M, x0, b)

    # Ensure that warnings are always reissued from this function
    warnings.filterwarnings('always', module='pyamg.krylov._srecg')

    # determine maxiter
    if maxiter is None:
        maxiter = int(1.3*len(b)) + 2
    elif maxiter < 1:
        raise ValueError('Number of iterations must be positive')

    # setup method
    r = b - A * x

    # precondition residual
    z = M * r
    res_norm = norm(r)

    # Append residual to list
    if residuals is not None:
        # z = M * r
        # precond_norm = np.inner(r.conjugate(), z)
        # precond_norm = np.sqrt(precond_norm)
        # residuals.append(precond_norm)
        residuals.append(res_norm)

    # Adjust tolerance
    # Check initial guess ( scaling by b, if b != 0,
    #   must account for case when norm(b) is very small)
    normb = norm(b)
    if normb == 0.0:
        normb = 1.0
    if res_norm < tol*normb:
        return (postprocess(x), 0)

    # Scale tol by ||r_0||_M
    if res_norm != 0.0:
        # precond_norm = np.inner(r.conjugate(), z)
        # precond_norm = np.sqrt(precond_norm)
        # tol = tol * precond_norm
        tol = tol * res_norm

    # Initialize list for previous search directions
    W_list = []

    # k = 0
    k = 0
    # while (res_norm > tol) and (k < maxiter):
    while True:
        # A-ortho the search directions for first iteration
        if k == 0:
            # W_0 = T(r_0)
            W = split_residual(z, t)
            # W_0 = A_orth(W_0)
            cgs(W, A)
        else:
            # W_k = A * W_{k-1}
            W = A * W
            # preconditioning step
            # W = M * W
            for i in range(t):
                W_temp = np.copy(W[:, i])
                np.ascontiguousarray(W_temp, dtype=W.dtype)
                W_temp = M * W_temp
                W[:, i] = W_temp
            W = bcgs(A, W_list, W)
            cgs(W, A)

        W_list.append(W)
        if len(W_list) > 2:
            del W_list[0]

        # alpha_k = W_k^T r_k
        alpha = W.conjugate().T.dot(r)

        # W * alpha
        W_alpha = W.dot(alpha)

        # x_k = X_k + W_k alpha_k
        x += W_alpha

        # r = r - A * W_k * alpha_k
        r -= A * W_alpha

        res_norm = norm(r)
        k += 1

        # Append residual to list
        if residuals is not None:
            # z = M * r
            # precond_norm = np.inner(r.conjugate(), z)
            # precond_norm = np.sqrt(precond_norm)
            # residuals.append(precond_norm)
            residuals.append(res_norm)

        if callback is not None:
            callback(x)

        # Check for convergence
        if res_norm < tol:
            return (postprocess(x), 0)

        if k == maxiter:
            return (postprocess(x), k)

# if __name__ == '__main__':
#    # from numpy import diag
#    # A = random((4,4))
#    # A = A*A.transpose() + diag([10,10,10,10])
#    # b = random((4,1))
#    # x0 = random((4,1))
#
#    from pyamg.gallery import stencil_grid
#    from numpy.random import random
#    A = stencil_grid([[0,-1,0],[-1,4,-1],[0,-1,0]],(100,100),
#                     dtype=float,format='csr')
#    b = random((A.shape[0],))
#    x0 = random((A.shape[0],))
#
#    import time
#    from scipy.sparse.linalg.isolve import cg as icg
#
#    print '\n\nTesting SRECG with %d x %d 2D Laplace Matrix' % \
#           (A.shape[0],A.shape[0])
#    t1=time.time()
#    (x,flag) = srecg(A,b,1,x0,tol=1e-8,maxiter=100)
#    t2=time.time()
#    print '%s took %0.3f ms' % ('srecg', (t2-t1)*1000.0)
#    print 'norm = %g'%(norm(b - A*x))
#    print 'info flag = %d'%(flag)
#
#    t1=time.time()
#    (y,flag) = icg(A,b,x0,tol=1e-8,maxiter=100)
#    t2=time.time()
#    print '\n%s took %0.3f ms' % ('linalg cg', (t2-t1)*1000.0)
#    print 'norm = %g'%(norm(b - A*y))
#    print 'info flag = %d'%(flag)
