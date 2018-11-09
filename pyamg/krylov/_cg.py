import numpy as np
from scipy.sparse.linalg.isolve.utils import make_system
from pyamg.util.linalg import norm
from warnings import warn


__all__ = ['cg']


def cg(A, b, x0=None, tol=1e-5, maxiter=None, xtype=None, M=None,
       callback=None, residuals=None):
    """Conjugate Gradient algorithm.

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
        relative convergence tolerance, i.e. tol is scaled by ||b||
        ||r_k|| < tol * ||b||
    maxiter : int
        maximum number of allowed iterations
    xtype : type
        dtype for the solution, default is automatic type detection
    M : array, matrix, sparse matrix, LinearOperator
        n x n, inverted preconditioner, i.e. solve M A x = M b.
    callback : function
        User-supplied function is called after each iteration as
        callback(xk), where xk is the current solution vector
    residuals : list
        residual history in the 2-norm, including the initial residual

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
    >>> from pyamg.krylov.cg import cg
    >>> from pyamg.util.linalg import norm
    >>> import numpy as np
    >>> from pyamg.gallery import poisson
    >>> A = poisson((10,10))
    >>> b = np.ones((A.shape[0],))
    >>> (x,flag) = cg(A,b, maxiter=2, tol=1e-8)
    >>> print norm(b - A*x)
    10.9370700187

    References
    ----------
    .. [1] Yousef Saad, "Iterative Methods for Sparse Linear Systems,
       Second Edition", SIAM, pp. 262-67, 2003
       http://www-users.cs.umn.edu/~saad/books.html

    """
    A, M, x, b, postprocess = make_system(A, M, x0, b)

    # Ensure that warnings are always reissued from this function
    import warnings
    warnings.filterwarnings('always', module='pyamg\.krylov\._cg')

    # determine maxiter
    if maxiter is None:
        maxiter = int(1.3*len(b)) + 2
    elif maxiter < 1:
        raise ValueError('Number of iterations must be positive')

    # setup method
    r = b - A*x
    z = M*r
    p = z.copy()
    rz = np.inner(r.conjugate(), z)

    normr = np.linalg.norm(r)

    if residuals is not None:
        residuals[:] = [normr]  # initial residual

    # Check initial guess ( scaling by b, if b != 0,
    #   must account for case when norm(b) is very small)
    normb = norm(b)
    if normb == 0.0:
        normb = 1.0
    if normr < tol*normb:
        return (postprocess(x), 0)

    # Scale tol by ||b||
    if normr != 0.0:
        tol = tol*normb

    # How often should r be recomputed
    recompute_r = 8

    iter = 0

    while True:
        Ap = A*p

        rz_old = rz
        # Step number in Saad's pseudocode
        pAp = np.inner(Ap.conjugate(), p)            # check curvature of A
        if pAp < 0.0:
            warn("\nIndefinite matrix detected in CG, aborting\n")
            return (postprocess(x), -1)

        alpha = rz/pAp                            # 3
        x += alpha * p                            # 4

        if np.mod(iter, recompute_r) and iter > 0:   # 5
            r -= alpha * Ap
        else:
            r = b - A*x

        z = M*r                                   # 6
        rz = np.inner(r.conjugate(), z)

        if rz <= 0.0:                              # check curvature of M
            if rz == 0.0:
                warn("\nSingular preconditioner detected in CG, aborting\n")
            else
                warn("\nIndefinite preconditioner detected in CG, aborting\n")
            return (postprocess(x), -1)

        beta = rz/rz_old                          # 7
        p *= beta                                 # 8
        p += z

        iter += 1

        normr = np.linalg.norm(r)

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
    import time
    from scipy.sparse.linalg.isolve import cg as icg
    import numpy as np

    A = stencil_grid([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], (100, 100),
                     dtype=float, format='csr')
    #b = random((A.shape[0],))
    xstar = 0*random((A.shape[0],))
    b = A * xstar
    x0 = random((A.shape[0],))


    print('\n\nTesting CG with {} x {} 2D Laplace Matrix'.format(A.shape[0],
                                                                 A.shape[0]))
    t1 = time.time()
    (x, flag) = cg(A, b, x0, tol=1e-8, maxiter=100)
    t2 = time.time()
    print('%s took %0.3f ms' % ('cg', (t2-t1)*1000.0))
    print('norm = %g' % (norm(b - A*x)))
    print('info flag = %d' % (flag))

    t1 = time.time()
    (y, flag) = icg(A, b, x0, tol=1e-8, maxiter=100)
    t2 = time.time()
    print('\n%s took %0.3f ms' % ('linalg cg', (t2-t1)*1000.0))
    print('norm = %g' % (norm(b - A*y)))
    print('info flag = %d' % (flag))

    print('-------------')
    criterion1 = []
    myerr = []
    norm = np.linalg.norm

    criterion1 = []
    criterion2 = []
    criterion5 = []
    backwarderror = []

    def mycb(xk):
        criterion1.append(norm(b - A * xk) / (norm(A.data)*norm(xk) + norm(b)))
        criterion2.append(norm(b - A * xk) / norm(b))
        criterion5.append(norm(b - A * xk) / norm(b - A * x0))
        backwarderror.append(norm(b - A * xk) / (norm(A.data*norm(xk) * norm(xstar))))

    t1 = time.time()
    res = []
    (x, flag) = cg(A, b, x0, tol=1e-8, maxiter=1000, callback=mycb)
    t2 = time.time()

    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams['text.usetex'] = True
    plt.semilogy(criterion1, label=r'$\frac{\|r_k\|}{\|b\| + \|A\|\|x_k\|}$')
    plt.semilogy(criterion2, label=r'$\frac{\|r_k\|}{\|b\|}$')
    plt.semilogy(criterion5, label=r'$\frac{\|r_k\|}{\|r_0\|}$')
    plt.semilogy(backwarderror, label=r'$\|e_k\|$')
    plt.legend()
    plt.show()
