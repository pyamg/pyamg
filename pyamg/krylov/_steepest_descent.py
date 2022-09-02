"""Steepest Descent projection method."""

import warnings
from warnings import warn
import numpy as np
from scipy import sparse
from ..util.linalg import norm
from ..util import make_system


def steepest_descent(A, b, x0=None, tol=1e-5, criteria='rr',
                     maxiter=None, M=None,
                     callback=None, residuals=None):
    """Steepest descent algorithm.

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
        Tolerance for stopping criteria
    criteria : string
        Stopping criteria, let r=r_k, x=x_k
        'rr':        ||r||       < tol ||b||
        'rr+':       ||r||       < tol (||b|| + ||A||_F ||x||)
        'MrMr':      ||M r||     < tol ||M b||
        'rMr':       <r, Mr>^1/2 < tol
        if ||b||=0, then set ||b||=1 for these tests.
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
    info : halting status of cg

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

    See Also
    --------
    _minimal_residual

    Examples
    --------
    >>> from pyamg.krylov import steepest_descent
    >>> from pyamg.util.linalg import norm
    >>> import numpy as np
    >>> from pyamg.gallery import poisson
    >>> A = poisson((10,10))
    >>> b = np.ones((A.shape[0],))
    >>> (x,flag) = steepest_descent(A,b, maxiter=2, tol=1e-8)
    >>> print(f'{norm(b - A*x):.6}')
    7.89436

    References
    ----------
    .. [1] Yousef Saad, "Iterative Methods for Sparse Linear Systems,
       Second Edition", SIAM, pp. 137--142, 2003
       http://www-users.cs.umn.edu/~saad/books.html

    """
    A, M, x, b, postprocess = make_system(A, M, x0, b)

    # Ensure that warnings are always reissued from this function
    warnings.filterwarnings('always', module='pyamg.krylov._steepest_descent')

    # determine maxiter
    if maxiter is None:
        maxiter = int(len(b))
    elif maxiter < 1:
        raise ValueError('Number of iterations must be positive')

    # setup method
    r = b - A @ x
    z = M @ r
    rz = np.inner(r.conjugate(), z)

    normr = np.linalg.norm(r)

    if residuals is not None:
        residuals[:] = [normr]  # initial residual

    # Check initial guess if b != 0,
    normb = norm(b)
    if normb == 0.0:
        normb = 1.0  # reset so that tol is unscaled

    # set the stopping criteria (see the docstring)
    if criteria == 'rr':
        rtol = tol * normb
    elif criteria == 'rr+':
        if sparse.issparse(A.A):
            normA = norm(A.A.data)
        elif isinstance(A.A, np.ndarray):
            normA = norm(np.ravel(A.A))
        else:
            raise ValueError('Unable to use ||A||_F with the current matrix format.')
        rtol = tol * (normA * np.linalg.norm(x) + normb)
    elif criteria == 'MrMr':
        normr = norm(z)
        normMb = norm(M @ b)
        rtol = tol * normMb
    elif criteria == 'rMr':
        normr = np.sqrt(rz)
        rtol = tol
    else:
        raise ValueError('Invalid stopping criteria.')

    # How often should r be recomputed
    recompute_r = 50

    it = 0

    while True:
        q = A @ z
        zAz = np.inner(z.conjugate(), q)                # check curvature of A
        if zAz < 0.0:
            warn('\nIndefinite matrix detected in steepest descent, aborting\n')
            return (postprocess(x), -1)

        alpha = rz / zAz                            # step size
        x = x + alpha * z

        it += 1
        if np.mod(it, recompute_r) and it > 0:
            r = b - A @ x
        else:
            r = r - alpha * q

        z = M @ r
        rz = np.inner(r.conjugate(), z)

        if rz < 0.0:                                # check curvature of M
            warn('\nIndefinite preconditioner detected in steepest descent, stopping.\n')
            return (postprocess(x), -1)

        normr = norm(r)

        if residuals is not None:
            residuals.append(normr)

        if callback is not None:
            callback(x)

        # set the stopping criteria (see the docstring)
        if criteria == 'rr':
            rtol = tol * normb
        elif criteria == 'rr+':
            rtol = tol * (normA * np.linalg.norm(x) + normb)
        elif criteria == 'MrMr':
            normr = norm(z)
            rtol = tol * normMb
        elif criteria == 'rMr':
            normr = np.sqrt(rz)
            rtol = tol

        if normr < rtol:
            return (postprocess(x), 0)

        if rz == 0.0:
            # important to test after testing normr < tol. rz == 0.0 is an
            # indicator of convergence when r = 0.0
            warn('\nSingular preconditioner detected in steepest descent, stopping.\n')
            return (postprocess(x), -1)

        if it == maxiter:
            return (postprocess(x), it)
