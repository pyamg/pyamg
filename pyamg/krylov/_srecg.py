"""Short Recurrence Enlarged Conjugate Gradient algorithm."""

from ._srecg_orthodir import srecg_orthodir
from ._srecg_orthodir_new import srecg_orthodir_new
from ._srecg_bcgs import srecg_bcgs


def srecg(A, b, x0=None, t=1, tol=1e-5, maxiter=None, M=None,
          callback=None, residuals=None, orthog='bcgs', **kwargs):
    """Short Recurrence Enlarged Conjugate Gradient (SRE-CG) algorithm.

    Solves the linear system Ax = b by dispatching to one of the SRE-CG
    variants selected by ``orthog``.

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
        Only supported for ``orthog='bcgs'``.
    callback : function
        User-supplied function is called after each iteration as
        ``callback(xk)``, where xk is the current solution vector.
    residuals : list
        Residual history in the 2-norm, including the initial residual.
    orthog : {'bcgs', 'orthodir', 'orthodir_new'}
        A-orthonormalization of the search directions; see
        :func:`srecg_bcgs`, :func:`srecg_orthodir` and
        :func:`srecg_orthodir_new`.
    **kwargs : dict
        Passed to the selected variant.

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

    References
    ----------
    .. [1] Grigori, Laura, Sophie Moufawad, and Frederic Nataf.
       "Enlarged Krylov Subspace Conjugate Gradient Methods for Reducing
       Communication", SIAM Journal on Matrix Analysis and Applications 37(2),
       pp. 744-773, 2016.

    Examples
    --------
    >>> from pyamg.krylov import srecg
    >>> from pyamg.util.linalg import norm
    >>> import numpy as np
    >>> from pyamg.gallery import poisson
    >>> A = poisson((10,10))
    >>> b = np.ones((A.shape[0],))
    >>> (x,flag) = srecg(A,b, t=4, maxiter=2, tol=1e-8)
    >>> print(f'{norm(b - A@x):.6}')
    10.3568

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
