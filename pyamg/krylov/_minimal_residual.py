"""Minimum Residual projection method."""

import warnings
from warnings import warn
import numpy as np
from ..util.linalg import norm
from ..util import make_system


def minimal_residual(A, b, x0=None, tol=1e-5,
                     maxiter=None, M=None,
                     callback=None, residuals=None):
    """Minimal residual (MR) algorithm. 1D projection method.

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
        Tolerance for stopping criteria, let r=r_k
        ||M r|| < tol ||M b||
        if ||b||=0, then set ||M b||=1 for these tests.
    maxiter : int
        maximum number of iterations allowed
    M : array, matrix, sparse matrix, LinearOperator
        n x n, inverted preconditioner, i.e. solve M A x = M b.
    callback : function
        User-supplied function is called after each iteration as
        callback(xk), where xk is the current solution vector
    residuals : list
        preconditioned residual history in the 2-norm,
        including the initial preconditioned residual

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
    The LinearOperator class is in scipy.sparse.linalg.
    Use this class if you prefer to define A or M as a mat-vec routine
    as opposed to explicitly constructing the matrix.

    ..
        minimal residual algorithm:      Preconditioned version:
        r = b - A x                      r = b - A x, z = M r
        while not converged:             while not converged:
            p = A r                          p = M A z
            alpha = (p,r) / (p,p)            alpha = (p, z) / (p, p)
            x = x + alpha r                  x = x + alpha z
            r = r - alpha p                  z = z - alpha p

    See Also
    --------
    _steepest_descent

    Examples
    --------
    >>> from pyamg.krylov import minimal_residual
    >>> from pyamg.util.linalg import norm
    >>> import numpy as np
    >>> from pyamg.gallery import poisson
    >>> A = poisson((10,10))
    >>> b = np.ones((A.shape[0],))
    >>> (x,flag) = minimal_residual(A,b, maxiter=2, tol=1e-8)
    >>> print(f'{norm(b - A*x):.6}')
    7.26369

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
    z = M @ r
    normr = norm(z)

    # store initial residual
    if residuals is not None:
        residuals[:] = [normr]

    # Check initial guess if b != 0,
    normb = norm(b)
    if normb == 0.0:
        normMb = 1.0  # reset so that tol is unscaled
    else:
        normMb = norm(M @ b)

    # set the stopping criteria (see the docstring)
    if normr < tol * normMb:
        return (postprocess(x), 0)

    # How often should r be recomputed
    recompute_r = 50

    it = 0

    while True:
        p = M @ (A @ z)

        # (p, z) = (M A M r, M r) = (M A z, z)
        pz = np.inner(p.conjugate(), z)  # check curvature of M^-1 A
        if pz < 0.0:
            warn('\nIndefinite matrix detected in minimal residual, stopping.\n')
            return (postprocess(x), -1)

        alpha = pz / np.inner(p.conjugate(), p)
        x = x + alpha * z

        it += 1

        if np.mod(it, recompute_r) and it > 0:
            r = b - A @ x
            z = M @ r
        else:
            z = z - alpha * p

        normr = norm(z)
        if residuals is not None:
            residuals.append(normr)

        if callback is not None:
            callback(x)

        # set the stopping criteria (see the docstring)
        if normr < tol * normMb:
            return (postprocess(x), 0)

        if it == maxiter:
            return (postprocess(x), it)
