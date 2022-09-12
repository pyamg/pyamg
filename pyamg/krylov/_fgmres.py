"""Flexible Generalized Minimum Residual Method (fGMRES) Krylov solver."""

import warnings
from warnings import warn
import numpy as np
from scipy.linalg import get_lapack_funcs
import scipy as sp
from ..util.linalg import norm
from ..util import make_system
from .. import amg_core


def _mysign(x):
    """Return complex sign of x."""
    if x == 0.0:
        return 1.0
    # return the complex "sign"
    return x/np.abs(x)


def fgmres(A, b, x0=None, tol=1e-5,
           restrt=None, maxiter=None,
           M=None, callback=None, residuals=None):
    """Flexible Generalized Minimum Residual Method (fGMRES).

    fGMRES iteratively refines the initial solution guess to the
    system Ax = b.  fGMRES is flexible in the sense that the right
    preconditioner (M) can vary from iteration to iteration.

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
        ||r|| < tol ||b||
        if ||b||=0, then set ||b||=1 for these tests.
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
        M need not be stationary for fgmres
    callback : function
        User-supplied function is called after each iteration as
        callback(xk), where xk is the current solution vector
    residuals : list
        residual history in the 2-norm, including the initial residual
    reorth : boolean
        If True, then a check is made whether to re-orthogonalize the Krylov
        space each GMRES iteration

    Returns
    -------
    xk, info
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

    fGMRES allows for non-stationary preconditioners, as opposed to GMRES

    For robustness, Householder reflections are used to orthonormalize
    the Krylov Space
    Givens Rotations are used to provide the residual norm each iteration
    Flexibility implies that the right preconditioner, M, can
    vary from iteration to iteration

    Examples
    --------
    >>> from pyamg.krylov import fgmres
    >>> from pyamg.util.linalg import norm
    >>> import numpy as np
    >>> from pyamg.gallery import poisson
    >>> A = poisson((10,10))
    >>> b = np.ones((A.shape[0],))
    >>> (x,flag) = fgmres(A,b, maxiter=2, tol=1e-8)
    >>> print(f'{norm(b - A*x):.6}')
    6.54282

    References
    ----------
    .. [1] Yousef Saad, "Iterative Methods for Sparse Linear Systems,
       Second Edition", SIAM, pp. 151-172, pp. 272-275, 2003
       http://www-users.cs.umn.edu/~saad/books.html

    """
    # Convert inputs to linear system, with error checking
    A, M, x, b, postprocess = make_system(A, M, x0, b)
    n = A.shape[0]

    # Ensure that warnings are always reissued from this function
    warnings.filterwarnings('always', module='pyamg.krylov._fgmres')

    # Get fast access to underlying BLAS routines
    [lartg] = get_lapack_funcs(['lartg'], [x])

    # Set number of outer and inner iterations
    # If no restarts,
    #     then set max_inner=maxiter and max_outer=n
    # If restarts are set,
    #     then set max_inner=restart and max_outer=maxiter
    if restrt:
        if maxiter:
            max_outer = maxiter
        else:
            max_outer = 1
        if restrt > n:
            warn('Setting restrt to maximum allowed, n.')
            restrt = n
        max_inner = restrt
    else:
        max_outer = 1
        if maxiter is None:
            maxiter = min(n, 40)
        elif maxiter > n:
            warn('Setting maxiter to maximum allowed, n.')
            maxiter = n
        max_inner = maxiter

    # Is this a one dimensional matrix?
    if n == 1:
        entry = np.ravel(A @ np.array([1.0], dtype=x.dtype))
        return (postprocess(b/entry), 0)

    # Prep for method
    r = b - A @ x

    normr = norm(r)
    if residuals is not None:
        residuals[:] = [normr]  # initial residual

    # Check initial guess if b != 0,
    normb = norm(b)
    if normb == 0.0:
        normb = 1.0   # reset so that tol is unscaled

    if normr < tol * normb:
        return (postprocess(x), 0)

    # Use separate variable to track iterations.  If convergence fails,
    # we cannot simply report niter = (outer-1)*max_outer + inner.  Numerical
    # error could cause the inner loop to halt while the actual ||r|| > tol.
    niter = 0

    # Begin fGMRES
    for _outer in range(max_outer):

        # Calculate vector w, which defines the Householder reflector
        #    Take shortcut in calculating,
        #    w = r + sign(r[1])*||r||_2*e_1
        w = r
        beta = _mysign(w[0]) * normr
        w[0] += beta
        w /= norm(w)

        # Preallocate for Krylov vectors, Householder reflectors and Hessenberg
        # matrix
        # Space required is O(n*max_inner)
        # Givens Rotations
        Q = np.zeros((4 * max_inner,), dtype=x.dtype)
        # upper Hessenberg matrix (made upper tri with Givens Rotations)
        H = np.zeros((max_inner, max_inner), dtype=x.dtype)
        W = np.zeros((max_inner, n), dtype=x.dtype)  # Householder reflectors
        # For fGMRES, preconditioned vectors must be stored
        # No Horner-like scheme exists that allow us to avoid this
        Z = np.zeros((n, max_inner), dtype=x.dtype)
        W[0, :] = w

        # Multiply r with (I - 2*w*w.T), i.e. apply the Householder reflector
        # This is the RHS vector for the problem in the Krylov Space
        g = np.zeros((n,), dtype=x.dtype)
        g[0] = -beta

        for inner in range(max_inner):
            # Calculate Krylov vector in two steps
            # (1) Calculate v = P_j = (I - 2*w*w.T)v, where k = inner
            v = -2.0 * np.conjugate(w[inner]) * w
            v[inner] += 1.0
            # (2) Calculate the rest, v = P_1*P_2*P_3...P_{j-1}*ej.
            # for j in range(inner-1,-1,-1):
            #    v = v - 2.0*dot(conjugate(W[j,:]), v)*W[j,:]
            amg_core.apply_householders(v, np.ravel(W), n, inner-1, -1, -1)

            # Apply preconditioner
            v = M @ v
            # Check for nan, inf
            # if isnan(v).any() or isinf(v).any():
            #    warn('inf or nan after application of preconditioner')
            #    return(postprocess(x), -1)
            Z[:, inner] = v

            # Calculate new search direction
            v = A @ v

            # Factor in all Householder orthogonal reflections on new search
            # direction
            # for j in range(inner+1):
            #    v = v - 2.0*dot(conjugate(W[j,:]), v)*W[j,:]
            amg_core.apply_householders(v, np.ravel(W), n, 0, inner+1, 1)

            # Calculate next Householder reflector, w
            #  w = v[inner+1:] + sign(v[inner+1])*||v[inner+1:]||_2*e_{inner+1)
            #  Note that if max_inner = n, then this is unnecessary for
            #  the last inner iteration, when inner = n-1.  Here we do
            #  not need to calculate a Householder reflector or Givens
            #  rotation because nnz(v) is already the desired length,
            #  i.e. we do not need to zero anything out.
            if inner != n-1:
                if inner < (max_inner-1):
                    w = W[inner+1, :]
                vslice = v[inner+1:]
                alpha = norm(vslice)
                if alpha != 0:
                    alpha = _mysign(vslice[0]) * alpha
                    # do not need the final reflector for future calculations
                    if inner < (max_inner-1):
                        w[inner+1:] = vslice
                        w[inner+1] += alpha
                        w /= norm(w)

                    # Apply new reflector to v
                    #  v = v - 2.0*w*(w.T*v)
                    v[inner+1] = -alpha
                    v[inner+2:] = 0.0

            if inner > 0:
                # Apply all previous Givens Rotations to v
                amg_core.apply_givens(Q, v, n, inner)

            # Calculate the next Givens rotation, where j = inner Note that if
            # max_inner = n, then this is unnecessary for the last inner
            # iteration, when inner = n-1.  Here we do not need to
            # calculate a Householder reflector or Givens rotation because
            # nnz(v) is already the desired length, i.e. we do not need to zero
            # anything out.
            if inner != n-1:
                if v[inner+1] != 0:
                    [c, s, r] = lartg(v[inner], v[inner+1])
                    Qblock = np.array([[c, s], [-np.conjugate(s), c]],
                                      dtype=x.dtype)
                    Q[(inner*4): ((inner+1)*4)] = np.ravel(Qblock).copy()

                    # Apply Givens Rotation to g, the RHS for the linear system
                    # in the Krylov Subspace.  Note that this dot does a matrix
                    # multiply, not an actual dot product where a conjugate
                    # transpose is taken
                    g[inner:inner+2] = np.dot(Qblock, g[inner:inner+2])

                    # Apply effect of Givens Rotation to v
                    v[inner] = np.dot(Qblock[0, :], v[inner:inner+2])
                    v[inner+1] = 0.0

            # Write to upper Hessenberg Matrix,
            #   the LHS for the linear system in the Krylov Subspace
            H[:, inner] = v[0:max_inner]

            # Don't update normr if last inner iteration, because
            # normr is calculated directly after this loop ends.
            if inner < max_inner-1:
                normr = np.abs(g[inner+1])
                if normr < tol * normb:
                    break
                if residuals is not None:
                    residuals.append(normr)

                if callback is not None:
                    y = sp.linalg.solve(H[0:(inner+1), 0:(inner+1)], g[0:(inner+1)])
                    update = np.dot(Z[:, 0:inner+1], y)
                    callback(x + update)

            niter += 1

        # end inner loop, back to outer loop

        # Find best update to x in Krylov Space, V.  Solve inner+1 x inner+1
        # system.  Apparently this is the best way to solve a triangular system
        # in the magical world of scipy
        # piv = arange(inner+1)
        # y = lu_solve((H[0:(inner+1),0:(inner+1)], piv),
        #              g[0:(inner+1)], trans=0)
        y = sp.linalg.solve(H[0:(inner+1), 0:(inner+1)], g[0:(inner+1)])

        # No Horner like scheme exists because the preconditioner can change
        # each iteration # Hence, we must store each preconditioned vector
        update = np.dot(Z[:, 0:inner+1], y)
        x = x + update
        r = b - A @ x

        # Allow user access to the iterates
        if callback is not None:
            callback(x)

        normr = norm(r)
        if residuals is not None:
            residuals.append(normr)

        # Has fGMRES stagnated?
        indices = (x != 0)
        if indices.any():
            change = np.max(np.abs(update[indices] / x[indices]))
            if change < 1e-12:
                # No change, halt
                return (postprocess(x), -1)

        # test for convergence
        if normr < tol * normb:
            return (postprocess(x), 0)

    # end outer loop

    return (postprocess(x), niter)
