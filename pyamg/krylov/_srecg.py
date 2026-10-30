"""Short Recurrence Enlarged Conjugate Gradient algorithm."""

from ._srecg_orthodir import srecg_orthodir
from ._srecg_orthodir_new import srecg_orthodir_new
from ._srecg_bcgs import srecg_bcgs


def srecg(A, b, x0=None, t=1, tol=1e-5, maxiter=None, M=None,
          callback=None, residuals=None, orthog='bcgs', **kwargs):
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
    # pass along **kwargs
    if orthog == 'orthodir':
        (x, flag) = srecg_orthodir(A, b, x0=x0, t=t, tol=tol, maxiter=maxiter,
                                   M=M, callback=callback,
                                   residuals=residuals, **kwargs)
    if orthog == 'orthodir_new':
        (x, flag) = srecg_orthodir_new(A, b, x0=x0, t=t, tol=tol, maxiter=maxiter,
                                       M=M, callback=callback,
                                       residuals=residuals, **kwargs)
    elif orthog == 'bcgs':
        (x, flag) = srecg_bcgs(A, b, x0=x0, t=t, tol=tol, maxiter=maxiter,
                               M=M, callback=callback,
                               residuals=residuals, **kwargs)
    return (x, flag)
