"""bicgstab Krylov solver."""

import warnings
import numpy as np
from scipy.sparse.linalg.isolve.utils import make_system
from scipy import sparse
from pyamg.util.linalg import norm


def bicgstab(A, b, x0=None, tol=1e-5, criteria='rr',
             maxiter=None, M=None,
             callback=None, residuals=None):
    """Biconjugate Gradient Algorithm with Stabilization.

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
    >>> from pyamg.krylov.bicgstab import bicgstab
    >>> from pyamg.util.linalg import norm
    >>> import numpy as np
    >>> from pyamg.gallery import poisson
    >>> A = poisson((10,10))
    >>> b = np.ones((A.shape[0],))
    >>> (x,flag) = bicgstab(A,b, maxiter=2, tol=1e-8)
    >>> print norm(b - A @ x)
    4.68163045309

    References
    ----------
    .. [1] Yousef Saad, "Iterative Methods for Sparse Linear Systems,
       Second Edition", SIAM, pp. 231-234, 2003
       http://www-users.cs.umn.edu/~saad/books.html

    """
    # Convert inputs to linear system, with error checking
    A, M, x, b, postprocess = make_system(A, M, x0, b)

    # Ensure that warnings are always reissued from this function
    warnings.filterwarnings('always', module='pyamg.krylov._bicgstab')

    # Check iteration numbers
    if maxiter is None:
        maxiter = len(x) + 5
    elif maxiter < 1:
        raise ValueError('Number of iterations must be positive')

    # Prep for method
    r = b - A @ x
    normr = norm(r)

    if residuals is not None:
        residuals[:] = [normr]

    # Check initial guess, if b != 0,
    normb = norm(b)
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
    else:
        raise ValueError('Invalid stopping criteria.')

    if normr < rtol:
        return (postprocess(x), 0)

    # Is thisAa one dimensional matrix?
    # Use a matvec to access A[0,0]
    if A.shape[0] == 1:
        entry = np.ravel(A @ np.array([1.0], dtype=x.dtype))
        return (postprocess(b/entry), 0)

    rstar = r.copy()
    p = r.copy()

    rrstarOld = np.inner(rstar.conjugate(), r)

    it = 0

    # Begin BiCGStab
    while True:
        Mp = M @ p
        AMp = A @ Mp

        # alpha = (r_j, rstar) / (A*M*p_j, rstar)
        alpha = rrstarOld/np.inner(rstar.conjugate(), AMp)

        # s_j = r_j - alpha*A*M*p_j
        s = r - alpha * AMp
        Ms = M @ s
        AMs = A @ Ms

        # omega = (A*M*s_j, s_j)/(A*M*s_j, A*M*s_j)
        omega = np.inner(AMs.conjugate(), s)/np.inner(AMs.conjugate(), AMs)

        # x_{j+1} = x_j +  alpha*M*p_j + omega*M*s_j
        x = x + alpha * Mp + omega * Ms

        # r_{j+1} = s_j - omega*A*M*s
        r = s - omega * AMs

        # beta_j = (r_{j+1}, rstar)/(r_j, rstar) * (alpha/omega)
        rrstarNew = np.inner(rstar.conjugate(), r)
        beta = (rrstarNew / rrstarOld) * (alpha / omega)
        rrstarOld = rrstarNew

        # p_{j+1} = r_{j+1} + beta*(p_j - omega*A*M*p)
        p = r + beta * (p - omega * AMp)

        it += 1

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

        if normr < rtol:
            return (postprocess(x), 0)

        if it == maxiter:
            return (postprocess(x), it)
