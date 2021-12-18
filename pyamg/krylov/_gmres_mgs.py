"""GMRES Gram-Schmidt-based implementation."""

import warnings
from warnings import warn

import numpy as np
import scipy as sp
from scipy.sparse.linalg.isolve.utils import make_system
from scipy.linalg import get_blas_funcs, get_lapack_funcs
from pyamg.util.linalg import norm


def apply_givens(Q, v, k):
    """Apply the first k Givens rotations in Q to v.

    Parameters
    ----------
    Q : list
        list of consecutive 2x2 Givens rotations
    v : array
        vector to apply the rotations to
    k : int
        number of rotations to apply.

    Returns
    -------
    v is changed in place

    Notes
    -----
    This routine is specialized for GMRES.  It assumes that the first Givens
    rotation is for dofs 0 and 1, the second Givens rotation is for
    dofs 1 and 2, and so on.

    """
    for j in range(k):
        Qloc = Q[j]
        v[j:j+2] = np.dot(Qloc, v[j:j+2])


def gmres_mgs(A, b, x0=None, tol=1e-5,
              restrt=None, maxiter=None,
              M=None, callback=None, residuals=None, reorth=False):
    """Generalized Minimum Residual Method (GMRES) based on MGS.

    GMRES iteratively refines the initial solution guess to the system
    Ax = b.  Modified Gram-Schmidt version.  Left preconditioning, leading
    to preconditioned residuals.

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
        preconditioned residual history in the 2-norm,
        including the initial preconditioned residual
    reorth : boolean
        If True, then a check is made whether to re-orthogonalize the Krylov
        space each GMRES iteration

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

    For robustness, modified Gram-Schmidt is used to orthogonalize the
    Krylov Space Givens Rotations are used to provide the residual norm
    each iteration

    The residual is the *preconditioned* residual.

    Examples
    --------
    >>> from pyamg.krylov import gmres
    >>> from pyamg.util.linalg import norm
    >>> import numpy as np
    >>> from pyamg.gallery import poisson
    >>> A = poisson((10,10))
    >>> b = np.ones((A.shape[0],))
    >>> (x,flag) = gmres(A,b, maxiter=2, tol=1e-8, orthog='mgs')
    >>> print norm(b - A*x)
    >>> 6.5428213057

    References
    ----------
    .. [1] Yousef Saad, "Iterative Methods for Sparse Linear Systems,
       Second Edition", SIAM, pp. 151-172, pp. 272-275, 2003
       http://www-users.cs.umn.edu/~saad/books.html

    .. [2] C. T. Kelley, http://www4.ncsu.edu/~ctk/matlab_roots.html

    """
    # Convert inputs to linear system, with error checking
    A, M, x, b, postprocess = make_system(A, M, x0, b)
    n = A.shape[0]

    # Ensure that warnings are always reissued from this function
    warnings.filterwarnings('always', module='pyamg.krylov._gmres_mgs')

    # Get fast access to underlying BLAS routines
    # dotc is the conjugate dot, dotu does no conjugation
    [lartg] = get_lapack_funcs(['lartg'], [x])
    if np.iscomplexobj(np.zeros((1,), dtype=x.dtype)):
        [axpy, dotu, dotc, scal] =\
            get_blas_funcs(['axpy', 'dotu', 'dotc', 'scal'], [x])
    else:
        # real type
        [axpy, dotu, dotc, scal] =\
            get_blas_funcs(['axpy', 'dot', 'dot', 'scal'], [x])

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
        if maxiter > n:
            warn('Setting maxiter to maximum allowed, n.')
            maxiter = n
        elif maxiter is None:
            maxiter = min(n, 40)
        max_inner = maxiter

    # Is this a one dimensional matrix?
    if n == 1:
        entry = np.ravel(A @ np.array([1.0], dtype=x.dtype))
        return (postprocess(b/entry), 0)

    # Prep for method
    r = b - A @ x

    # Apply preconditioner
    r = M @ r

    normr = norm(r)
    if residuals is not None:
        residuals[:] = [normr]  # initial residual

    # Check initial guess if b != 0,
    normb = norm(b)
    if normb == 0.0:
        normMb = 1.0   # reset so that tol is unscaled
    else:
        normMb = norm(M @ b)

    # set the stopping criteria (see the docstring)
    if normr < tol * normMb:
        return (postprocess(x), 0)

    # Use separate variable to track iterations.  If convergence fails, we
    # cannot simply report niter = (outer-1)*max_outer + inner.  Numerical
    # error could cause the inner loop to halt while the actual ||r|| > tolerance.
    niter = 0

    # Begin GMRES
    for _outer in range(max_outer):

        # Preallocate for Givens Rotations, Hessenberg matrix and Krylov Space
        # Space required is O(n*max_inner).
        # NOTE:  We are dealing with row-major matrices, so we traverse in a
        #        row-major fashion,
        #        i.e., H and V's transpose is what we store.
        Q = []  # Givens Rotations
        # Upper Hessenberg matrix, which is then
        #   converted to upper tri with Givens Rots
        H = np.zeros((max_inner+1, max_inner+1), dtype=x.dtype)
        V = np.zeros((max_inner+1, n), dtype=x.dtype)  # Krylov Space
        # vs store the pointers to each column of V.
        #   This saves a considerable amount of time.
        vs = []
        # v = r/normr
        V[0, :] = scal(1.0/normr, r)
        vs.append(V[0, :])

        # This is the RHS vector for the problem in the Krylov Space
        g = np.zeros((n,), dtype=x.dtype)
        g[0] = normr

        for inner in range(max_inner):

            # New Search Direction
            v = V[inner+1, :]
            v[:] = np.ravel(M @ (A @ vs[-1]))
            vs.append(v)
            normv_old = norm(v)

            #  Modified Gram Schmidt
            for k in range(inner+1):
                vk = vs[k]
                alpha = dotc(vk, v)
                H[inner, k] = alpha
                v[:] = axpy(vk, v, n, -alpha)

            normv = norm(v)
            H[inner, inner+1] = normv

            # Re-orthogonalize
            if (reorth is True) and (normv_old == normv_old + 0.001 * normv):
                for k in range(inner+1):
                    vk = vs[k]
                    alpha = dotc(vk, v)
                    H[inner, k] = H[inner, k] + alpha
                    v[:] = axpy(vk, v, n, -alpha)

            # Check for breakdown
            if H[inner, inner+1] != 0.0:
                v[:] = scal(1.0/H[inner, inner+1], v)

            # Apply previous Givens rotations to H
            if inner > 0:
                apply_givens(Q, H[inner, :], inner)

            # Calculate and apply next complex-valued Givens Rotation
            # for the last inner iteration, when inner = n-1.
            # ==> Note that if max_inner = n, then this is unnecessary
            if inner != n-1:
                if H[inner, inner+1] != 0:
                    [c, s, r] = lartg(H[inner, inner], H[inner, inner+1])
                    Qblock = np.array([[c, s], [-np.conjugate(s), c]], dtype=x.dtype)
                    Q.append(Qblock)

                    # Apply Givens Rotation to g,
                    #   the RHS for the linear system in the Krylov Subspace.
                    g[inner:inner+2] = np.dot(Qblock, g[inner:inner+2])

                    # Apply effect of Givens Rotation to H
                    H[inner, inner] = dotu(Qblock[0, :], H[inner, inner:inner+2])
                    H[inner, inner+1] = 0.0

            niter += 1

            # Do not update normr if last inner iteration, because
            # normr is calculated directly after this loop ends.
            if inner < max_inner-1:
                normr = np.abs(g[inner+1])
                if normr < tol * normMb:
                    break

                if residuals is not None:
                    residuals.append(normr)

                if callback is not None:
                    y = sp.linalg.solve(H[0:inner+1, 0:inner+1].T, g[0:inner+1])
                    update = np.ravel(V[:inner+1, :].T.dot(y.reshape(-1, 1)))
                    callback(x + update)

        # end inner loop, back to outer loop

        # Find best update to x in Krylov Space V.  Solve inner x inner system.
        y = sp.linalg.solve(H[0:inner+1, 0:inner+1].T, g[0:inner+1])
        update = np.ravel(V[:inner+1, :].T.dot(y.reshape(-1, 1)))
        x = x + update
        r = b - A @ x

        # Apply preconditioner
        r = M @ r
        normr = norm(r)

        # Allow user access to the iterates
        if callback is not None:
            callback(x)

        if residuals is not None:
            residuals.append(normr)

        # Has GMRES stagnated?
        indices = (x != 0)
        if indices.any():
            change = np.max(np.abs(update[indices] / x[indices]))
            if change < 1e-12:
                # No change, halt
                return (postprocess(x), -1)

        # test for convergence
        if normr < tol * normMb:
            return (postprocess(x), 0)

    # end outer loop

    return (postprocess(x), niter)
