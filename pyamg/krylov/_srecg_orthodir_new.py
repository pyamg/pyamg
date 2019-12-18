import numpy as np
from scipy.linalg import get_blas_funcs
from scipy.sparse.linalg.isolve.utils import make_system
from scipy.linalg import fractional_matrix_power
from pyamg.util.linalg import norm, BCGS, CGS, split_residual
from warnings import warn


__all__ = ['srecg_orthodir_new']


def srecg_orthodir_new(A, b, x0=None, t=1, tol=1e-5, maxiter=None, xtype=None, M=None,
       callback=None, residuals=None, **kwargs):
    '''Short Recurrence Enlarged Conjugate Gradient algorithm

    ****** LEFT PRECONDITIONING NOT SUPPORTED CURRENTLY ********

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
    
    '''
    A, M, x, b, postprocess = make_system(A, M, x0, b)

    # Ensure that warnings are always reissued from this function
    import warnings
    warnings.filterwarnings('always', module='pyamg.krylov._srecg')

    # determine maxiter
    if maxiter is None:
        maxiter = int(1.3*len(b)) + 2
    elif maxiter < 1:
        raise ValueError('Number of iterations must be positive')
    
    # setup method
    r = b - A * x

    # precondition residual
    #z = M * r
    res_norm = norm(r)

    # Append residual to list
    if residuals is not None:
        #z = M * r
        #precond_norm = np.inner(r.conjugate(), z)
        #precond_norm = np.sqrt(precond_norm)
        #residuals.append(precond_norm)
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
        #precond_norm = np.inner(r.conjugate(), z)
        #precond_norm = np.sqrt(precond_norm)
        #tol = tol * precond_norm
        tol = tol * res_norm

    # Initialize list for previous search directions
    W_list = []

    # k = 0
    k = 0

    # Initial search directions
    W = split_residual(r, t)
    W_list.append(np.zeros(W.shape))
    CGS(W, A)

    # grab blas function to be used later for search directions solve
    L = np.zeros((t, t), dtype=W.dtype) 
    dtrsm = get_blas_funcs(['trsm'], [W, L])[0]

    AW = A * W
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

        #r = r - A * W_k * alpha_k
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
            return (postprocess(x), 0)

        if k == maxiter:
            return (postprocess(x), k)

        # Update search directions
        AW1 = AW_list[1]
        AW2 = AW_list[0]

        gamma = AW1.conjugate().T.dot(AW1)
        rho = AW2.conjugate().T.dot(AW1)
        P = AW1 - W_list[1].dot(gamma) - W_list[0].dot(rho)

        # Do Cholesky of P^T A P
        Z = np.linalg.cholesky(P.conjugate().T.dot(A * P))
        # Solve upper triangular system for W 
        W = dtrsm(1.0, Z.T, P, side=1, lower=1, diag=0)
        # Solve upper triangular system for AW
        AW = dtrsm(1.0, Z.T, A*P, side=1, lower=1, diag=0)
