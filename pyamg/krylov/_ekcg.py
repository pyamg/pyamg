"""Short Recurrence Enlarged Conjugate Gradient algorithm."""
import warnings

import numpy as np

from ..util.linalg import norm, split_residual
from ..util import make_system


def ekcg(A, b, x0=None, t=1, tol=1e-5, maxiter=None, M=None,
         callback=None, res_callback=None, residuals=None):
    """Enlarged Krylov Conjugate Gradient (EK-CG) algorithm.

    Solves the linear system Ax = b. Preconditioning is supported.

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
    M : array, matrix, sparse matrix, LinearOperator
        Preconditioner applied to the residual as ``M @ r``.  M should be
        Hermitian positive definite.
    callback : function
        User-supplied function is called after each iteration as
        ``callback(X)``, where X is the (n,t) block of partial solutions;
        the current solution is ``X.sum(axis=1)``.
    res_callback : function
        User-supplied function is called after each iteration as
        ``res_callback(R)``, where R is the (n,t) block of split residuals.
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
    The LinearOperator class is in scipy.sparse.linalg.
    Use this class if you prefer to define A or M as a mat-vec routine
    as opposed to explicitly constructing the matrix.

    References
    ----------
    .. [1] Grigori, Laura, Sophie Moufawad, and Frederic Nataf.
       "Enlarged Krylov Subspace Conjugate Gradient Methods for Reducing
       Communication", SIAM Journal on Matrix Analysis and Applications 37(2),
       pp. 744-773, 2016.

    Examples
    --------
    >>> from pyamg.krylov import ekcg
    >>> from pyamg.util.linalg import norm
    >>> import numpy as np
    >>> from pyamg.gallery import poisson
    >>> A = poisson((10,10))
    >>> b = np.ones((A.shape[0],))
    >>> (x,flag) = ekcg(A,b, t=4, maxiter=2, tol=1e-8)
    >>> print(f'{norm(b - A@x):.6}')
    10.3568

    """
    A, M, x, b = make_system(A, M, x0, b)

    # Ensure that warnings are always reissued from this function
    warnings.filterwarnings('always', module='pyamg.krylov._ekcg')

    # determine maxiter
    if maxiter is None:
        maxiter = int(1.3*len(b)) + 2
    elif maxiter < 1:
        raise ValueError('Number of iterations must be positive')

    # setup method
    r = b - A @ x
    res_norm = norm(r)

    # Append residual to list
    if residuals is not None:
        residuals.append(res_norm)

    # Adjust tolerance
    # Check initial guess ( scaling by b, if b != 0,
    #   must account for case when norm(b) is very small)
    normb = norm(b)
    if normb == 0.0:
        normb = 1.0
    if res_norm < tol*normb:
        return (x, 0)

    # Scale tol by ||r_0||
    if res_norm != 0.0:
        tol = tol * res_norm

    # Split residual
    R = split_residual(r, t)

    # Initialize X to be 0
    X = np.zeros_like(R)
    for i in range(t):
        X[:, i] = x

    # First check for callback functions
    if callback is not None:
        # x = np.sum(X, axis=1)
        callback(X)

    if res_callback is not None:
        # r = np.sum(R, axis=1)
        res_callback(R)

    # Precondition residual
    Z = M @ R

    # Initialize P and P_{k-1}
    P = np.zeros_like(Z, A.dtype)
    P_1 = np.zeros_like(Z, A.dtype)

    # Initialize A @ P and A @ P_{k-1}
    AP = np.zeros_like(Z, A.dtype)
    AP_1 = np.zeros_like(Z, A.dtype)

    # Iterations variable
    k = 1
    while True:
        # P_k = Z_k (Z_k^T A Z_k)^1/2
        AZ = A @ Z
        ZAZ = Z.conjugate().T.dot(AZ)

        u, s, vh = np.linalg.svd(ZAZ, full_matrices=True)
        uh = u.conjugate().T
        s_diag = np.diag(1.0/np.sqrt(s))
        P = Z.dot(vh.T.dot(s_diag.dot(uh)))
        AP = A @ P

        # alpha_k = P_k^T R_{k-1}
        alpha = P.conjugate().T.dot(R)

        # P_k * alpha_k
        P_alpha = P.dot(alpha)

        # X_k = X_{k-1} + P_k * alpha_k
        X += P_alpha

        # R_k = R_{k-1} - A @ P_k * alpha_k
        R -= AP.dot(alpha)

        # Sum columns of R
        r = np.sum(R, axis=1)
        res_norm = norm(r)

        # Append residual to list
        if residuals is not None:
            residuals.append(res_norm)

        if callback is not None:
            # x = np.sum(X, axis=1)
            callback(X)

        if res_callback is not None:
            # r = np.sum(R, axis=1)
            res_callback(R)

        # Check for convergence
        if res_norm < tol:
            x = np.sum(X, axis=1)
            return (x, 0)

        if k == maxiter:
            x = np.sum(X, axis=1)
            return (x, k)

        # Update iteration
        k += 1

        # Precondition AP
        MAP = M @ AP

        # Orthodir A-orthonormalization
        # gamma = (A P_k)^T * (M A P_k)
        gamma = AP.conjugate().T.dot(MAP)

        # rho = (A P_{k-1})^T * (M A P_k)
        rho = AP_1.conjugate().T.dot(MAP)

        # Z_{k+1} = M A P_k - P_k * gamma - P_{k-1} * rho
        Z = MAP - P.dot(gamma) - P_1.dot(rho)

        # Update P_1 and AP_1
        P_1 = np.copy(P)
        AP_1 = np.copy(AP)
