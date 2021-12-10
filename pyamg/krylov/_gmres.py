"""Generalized Minimum Residual Method (GMRES) Krylov solver."""

from ._gmres_mgs import gmres_mgs
from ._gmres_householder import gmres_householder


def gmres(A, b, x0=None, tol=1e-5, restrt=None, maxiter=None,
          M=None, callback=None, residuals=None, orthog='householder',
          **kwargs):
    """Generalized Minimum Residual Method (GMRES).

    GMRES iteratively refines the initial solution guess to the
    system Ax = b.  Left preconditioned.  Residuals are preconditioned residuals.

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
           ||M r||     < tol ||M b||
        if ||b||=0, then set ||M b||=1 for these tests.
    restrt : None, int
        - if int, restrt is max number of inner iterations
          and maxiter is the max number of outer iterations
        - if None, do not restart GMRES, and max number of inner iterations
          is maxiter
    maxiter : None, int
        - if restrt is None, maxiter is the max number of inner iterations
          and GMRES does not restart
        - if restrt is int, maxiter is the max number of outer iterations,
          and restrt is the max number of inner iterations
        - defaults to min(n,40) if restart=None
    M : array, matrix, sparse matrix, LinearOperator
        n x n, inverted preconditioner, i.e. solve M A x = M b.
    callback : function
        User-supplied function is called after each iteration as
        callback(xk), where xk is the current solution vector
    residuals : list
        preconditioned residual history in the 2-norm, including the initial residual
    orthog : string
        'householder' calls _gmres_householder which uses Householder
        reflections to find the orthogonal basis for the Krylov space.
        'mgs' calls _gmres_mgs which uses modified Gram-Schmidt to find the
        orthogonal basis for the Krylov space

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

    The orthogonalization method, orthog='householder', is more robust
    than orthog='mgs', however for the majority of problems your
    problem will converge before 'mgs' loses orthogonality in your basis.

    orthog='householder' has been more rigorously tested, and is
    therefore currently the default

    The residual is the *preconditioned* residual.


    Examples
    --------
    >>> from pyamg.krylov import gmres
    >>> from pyamg.util.linalg import norm
    >>> import numpy as np
    >>> from pyamg.gallery import poisson
    >>> A = poisson((10,10))
    >>> b = np.ones((A.shape[0],))
    >>> (x,flag) = gmres(A,b, maxiter=2, tol=1e-8)
    >>> print norm(b - A*x)
    6.5428213057

    References
    ----------
    .. [1] Yousef Saad, "Iterative Methods for Sparse Linear Systems,
       Second Edition", SIAM, pp. 151-172, pp. 272-275, 2003
       http://www-users.cs.umn.edu/~saad/books.html

    """
    # pass along **kwargs
    if orthog == 'householder':
        (x, flag) = gmres_householder(A, b, x0=x0, tol=tol, restrt=restrt,
                                      maxiter=maxiter, M=M,
                                      callback=callback, residuals=residuals,
                                      **kwargs)
    elif orthog == 'mgs':
        (x, flag) = gmres_mgs(A, b, x0=x0, tol=tol, restrt=restrt,
                              maxiter=maxiter, M=M,
                              callback=callback, residuals=residuals, **kwargs)

    return (x, flag)
