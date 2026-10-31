"""Short recurrence Enlarged CG."""
import warnings

import numpy as np
from scipy.linalg import get_blas_funcs
from scipy.linalg import lapack

from ..util.linalg import norm, cgs, split_residual
from ..util import make_system


def srecg_orthodir_new(A, b, x0=None, t=1, tol=1e-5, maxiter=None, M=None,
                       callback=None, residuals=None):
    """SRE-CG with Orthodir A-orthonormalization via Cholesky.

    Solves the linear system Ax = b. Preconditioning is not supported.
    Same as :func:`srecg_orthodir`, but A-orthonormalizes with a Cholesky
    factorization of Z^H A Z and triangular solves.

    Parameters
    ----------
    A : array, matrix, sparse matrix, LinearOperator
        Linear system of size (n,n) to solve.
    b : array, matrix
        Right hand side of size (n,) or (n,1).
    x0 : array, matrix
        Initial guess, default is a vector of zeros.
    t : int
        Number of subdomains the residual is split into.
    tol : float
        Relative tolerance.  The solve stops when ||r_k|| < tol ||r_0||
        in the 2-norm.
    maxiter : int
        Maximum number of iterations allowed.
    M : None
        Preconditioning is not supported; passing M raises ValueError.
    callback : function
        User-supplied function is called after each iteration as
        ``callback(xk)``, where xk is the current solution vector.
    residuals : list
        Residual history in the 2-norm, including the initial residual.

    Returns
    -------
    array
        Updated guess after k iterations to the solution of Ax = b.
    int
        Return status

            ==  =======================================
            0   successful exit
            >0  convergence to tolerance not achieved,
                return iteration count instead.
            ==  =======================================

    Notes
    -----
    Experimental: for t > 1 this variant can break down (NaN residuals)
    where :func:`srecg_orthodir` converges.

    The LinearOperator class is in scipy.sparse.linalg.
    Use this class if you prefer to define A as a mat-vec routine
    as opposed to explicitly constructing the matrix.

    References
    ----------
    .. [1] Grigori, Laura, Sophie Moufawad, and Frederic Nataf.
       "Enlarged Krylov Subspace Conjugate Gradient Methods for Reducing
       Communication", SIAM Journal on Matrix Analysis and Applications 37(2),
       pp. 744-773, 2016.

    Examples
    --------
    >>> from pyamg.krylov import srecg_orthodir_new
    >>> from pyamg.util.linalg import norm
    >>> import numpy as np
    >>> from pyamg.gallery import poisson
    >>> A = poisson((10,10))
    >>> b = np.ones((A.shape[0],))
    >>> (x,flag) = srecg_orthodir_new(A,b, t=4, maxiter=2, tol=1e-8)
    >>> print(f'{norm(b - A@x):.6}')
    10.8625

    """
    if M is not None:
        raise ValueError('Left preconditioning for ECG not supported.')

    A, M, x, b = make_system(A, M, x0, b)

    # Ensure that warnings are always reissued from this function
    warnings.filterwarnings('always', module='pyamg.krylov._srecg')

    # determine maxiter
    if maxiter is None:
        maxiter = int(1.3*len(b)) + 2
    elif maxiter < 1:
        raise ValueError('Number of iterations must be positive')

    # setup method
    r = b - A @ x

    # precondition residual
    # z = M @ r
    res_norm = norm(r)

    # Append residual to list
    if residuals is not None:
        # z = M @ r
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
        return (x, 0)

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

    # Initial search directions
    W = split_residual(r, t)
    W_list.append(np.zeros(W.shape))
    cgs(W, A)

    # grab blas function to be used later for search directions solve
    L = np.zeros((t, t), dtype=W.dtype)
    dtrsm = get_blas_funcs(['trsm'], [W, L])[0]

    AW = A @ W
    AW_list = []
    AW_list.append(np.zeros(AW.shape))
    while True:
        # Append W to previous search directions
        W_list.append(W)
        AW_list.append(AW)
        if len(W_list) > 2:
            del W_list[0]
            del AW_list[0]

        # alpha_k = W_k^T r_k
        alpha = W.conjugate().T.dot(r)

        # W * alpha
        W_alpha = W.dot(alpha)

        # x_k = X_k + W_k alpha_k
        x += W_alpha

        # r = r - A @ W_k * alpha_k
        r -= AW.dot(alpha)

        res_norm = norm(r)
        k += 1

        # Append residual to list
        if residuals is not None:
            residuals.append(res_norm)

        if callback is not None:
            callback(x)

        # Check for convergence
        if res_norm < tol:
            return (x, 0)

        if k == maxiter:
            return (x, k)

        # Update search directions
        AW1 = AW_list[1]
        AW2 = AW_list[0]

        gamma = AW1.conjugate().T.dot(AW1)
        rho = AW2.conjugate().T.dot(AW1)
        P = AW1 - W_list[1].dot(gamma) - W_list[0].dot(rho)

        AP = A @ P
        # Do Cholesky of P^T A P
        # Z = np.linalg.cholesky(P.conjugate().T.dot(AP))
        Z = lapack.dpotrf(P.conjugate().T.dot(AP))[0]  # pylint: disable=no-member

        # Solve upper triangular system for W
        W = dtrsm(1.0, Z.T, P, side=1, lower=1, diag=0)
        # Solve upper triangular system for AW
        AW = dtrsm(1.0, Z.T, AP, side=1, lower=1, diag=0)
