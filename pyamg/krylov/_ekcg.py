"""Short Recurrence Enlarged Conjugate Gradient algorithm."""
import warnings

import numpy as np

from ..util.linalg import norm, split_residual
from ..util import make_system


def ekcg(A, b, x0=None, t=1, tol=1e-5, maxiter=None, M=None,
         callback=None, res_callback=None, residuals=None):
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
    >>> from pyamg.krylov.ekcg import ekcg
    >>> from pyamg.util.linalg import norm
    >>> import numpy as np
    >>> from pyamg.gallery import poisson
    >>> A = poisson((10,10))
    >>> b = np.ones((A.shape[0],))
    >>> (x,flag) = ekcg(A,b, maxiter=2, tol=1e-8)
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
    warnings.filterwarnings('always', module='pyamg.krylov._ekcg')

    # determine maxiter
    if maxiter is None:
        maxiter = int(1.3*len(b)) + 2
    elif maxiter < 1:
        raise ValueError('Number of iterations must be positive')

    # setup method
    r = b - A * x
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
        return (postprocess(x), 0)

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
    Z = np.zeros_like(R)
    for i in range(R.shape[1]):
        r = np.copy(R[:, i])
        Z[:, i] = M * r
    # Z = M * R

    # Initialize P and P_{k-1}
    P = np.zeros_like(Z, A.dtype)
    P_1 = np.zeros_like(Z, A.dtype)

    # Initialize A * P and A * P_{k-1}
    AP = np.zeros_like(Z, A.dtype)
    AP_1 = np.zeros_like(Z, A.dtype)

    # Iterations variable
    k = 1
    while True:
        # P_k = Z_k (Z_k^T A Z_k)^1/2
        AZ = A * Z
        ZAZ = Z.conjugate().T.dot(AZ)

        u, s, vh = np.linalg.svd(ZAZ, full_matrices=True)
        uh = u.conjugate().T
        s_diag = np.diag(1.0/np.sqrt(s))
        P = Z.dot(vh.T.dot(s_diag.dot(uh)))
        AP = A * P

        # alpha_k = P_k^T R_{k-1}
        alpha = P.conjugate().T.dot(R)

        # P_k * alpha_k
        P_alpha = P.dot(alpha)

        # X_k = X_{k-1} + P_k * alpha_k
        X += P_alpha

        # R_k = R_{k-1} - A * P_k * alpha_k
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
            return (postprocess(x), 0)

        if k == maxiter:
            x = np.sum(X, axis=1)
            return (postprocess(x), k)

        # Update iteration
        k += 1

        # Precondition AP
        MAP = np.zeros_like(AP)
        for i in range(AP.shape[1]):
            ap = np.copy(AP[:, i])
            MAP[:, i] = M * ap
        # MAP = M * AP

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
