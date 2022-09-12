"""Conjugate Gradient, Normal Residual Krylov solver."""

import warnings
from warnings import warn
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import aslinearoperator
from ..util.linalg import norm
from ..util import make_system


def cgnr(A, b, x0=None, tol=1e-5, criteria='rr',
         maxiter=None, M=None,
         callback=None, residuals=None):
    """Conjugate Gradient, Normal Residual algorithm.

    Applies CG to the normal equations, A.H A x = A.H b. Left preconditioning
    is supported.  Note that unless A is well-conditioned, the use of
    CGNR is inadvisable

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
        n x n, inverted preconditioner, i.e. solve M A.H A x = M A.H b.
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
    The LinearOperator class is in scipy.sparse.linalg.
    Use this class if you prefer to define A or M as a mat-vec routine
    as opposed to explicitly constructing the matrix.

    Examples
    --------
    >>> from pyamg.krylov import cgnr
    >>> from pyamg.util.linalg import norm
    >>> import numpy as np
    >>> from pyamg.gallery import poisson
    >>> A = poisson((10,10))
    >>> b = np.ones((A.shape[0],))
    >>> (x,flag) = cgnr(A,b, maxiter=2, tol=1e-8)
    >>> print(f'{norm(b - A*x):.6}')
    9.39102

    References
    ----------
    .. [1] Yousef Saad, "Iterative Methods for Sparse Linear Systems,
       Second Edition", SIAM, pp. 276-7, 2003
       http://www-users.cs.umn.edu/~saad/books.html

    """
    # Store the conjugate transpose explicitly as it will be used much later on
    if sparse.isspmatrix(A):
        AH = A.H
    else:
        # avoid doing this since A may be a different sparse type
        AH = aslinearoperator(np.asarray(A).conj().T)

    # Convert inputs to linear system, with error checking
    A, M, x, b, postprocess = make_system(A, M, x0, b)
    n = A.shape[0]

    # Ensure that warnings are always reissued from this function
    warnings.filterwarnings('always', module='pyamg.krylov._cgnr')

    # How often should r be recomputed
    recompute_r = 8

    # Check iteration numbers. CGNR suffers from loss of orthogonality quite
    # easily, so we arbitrarily let the method go up to 130% over the
    # theoretically necessary limit of maxiter=n
    if maxiter is None:
        maxiter = int(np.ceil(1.3*n)) + 2
    elif maxiter < 1:
        raise ValueError('Number of iterations must be positive')
    elif maxiter > (1.3*n):
        warn('maximum allowed inner iterations (maxiter) are the 130% times \
              the number of dofs')
        maxiter = int(np.ceil(1.3*n)) + 2

    # Prep for method
    r = b - A @ x
    rhat = AH @ r
    normr = norm(r)

    # Apply preconditioner and calculate initial search direction
    z = M @ rhat
    p = z.copy()
    old_zr = np.inner(z.conjugate(), rhat)

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
        normr = np.sqrt(old_zr)
        rtol = tol
    else:
        raise ValueError('Invalid stopping criteria.')

    if normr < rtol:
        return (postprocess(x), 0)

    # Begin CGNR

    it = 0

    while True:
        # w_j = A p_j
        w = A @ p

        # alpha = (z_j, rhat_j) / (w_j, w_j)
        alpha = old_zr / np.inner(w.conjugate(), w)

        # x_{j+1} = x_j + alpha*p_j
        x += alpha * p

        # r_{j+1} = r_j - alpha*w_j
        if np.mod(it, recompute_r) and it > 0:
            r -= alpha * w
        else:
            r = b - A @ x

        # rhat_{j+1} = A.H*r_{j+1}
        rhat = AH @ r

        # z_{j+1} = M*r_{j+1}
        z = M @ rhat

        # beta = (z_{j+1}, rhat_{j+1}) / (z_j, rhat_j)
        new_zr = np.inner(z.conjugate(), rhat)
        beta = new_zr / old_zr
        old_zr = new_zr

        # p_{j+1} = A.H*z_{j+1} + beta*p_j
        p *= beta
        p += z

        it += 1
        normr = np.linalg.norm(r)

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
            normr = np.sqrt(new_zr)
            rtol = tol

        if normr < rtol:
            return (postprocess(x), 0)

        if it == maxiter:
            return (postprocess(x), it)
