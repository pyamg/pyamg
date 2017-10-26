from __future__ import print_function
import numpy as np
import scipy as sp
from scipy.sparse.linalg.isolve.utils import make_system
from scipy.sparse.sputils import upcast
from scipy.linalg import get_blas_funcs, get_lapack_funcs
from warnings import warn


__all__ = ['gmres_mgs']


def apply_givens(Q, v, k):
    '''
    Apply the first k Givens rotations in Q to v

    Parameters
    ----------
    Q : {list}
        list of consecutive 2x2 Givens rotations
    v : {array}
        vector to apply the rotations to
    k : {int}
        number of rotations to apply.

    Returns
    -------
    v is changed in place

    Notes
    -----
    This routine is specialized for GMRES.  It assumes that the first Givens
    rotation is for dofs 0 and 1, the second Givens rotation is for
    dofs 1 and 2, and so on.
    '''

    for j in range(k):
        Qloc = Q[j]
        v[j:j+2] = np.dot(Qloc, v[j:j+2])


def gmres_mgs(A, b, x0=None, tol=1e-5, restrt=None, maxiter=None, xtype=None,
              M=None, callback=None, residuals=None, reorth=False):
    '''
    Generalized Minimum Residual Method (GMRES)
        GMRES iteratively refines the initial solution guess to the system
        Ax = b
        Modified Gram-Schmidt version

    Parameters
    ----------
    A : {array, matrix, sparse matrix, LinearOperator}
        n x n, linear system to solve
    b : {array, matrix}
        right hand side, shape is (n,) or (n,1)
    x0 : {array, matrix}
        initial guess, default is a vector of zeros
    tol : float
        relative convergence tolerance, i.e. tol is scaled by the norm
        of the initial preconditioned residual
    restrt : {None, int}
        - if int, restrt is max number of inner iterations
          and maxiter is the max number of outer iterations
        - if None, do not restart GMRES, and max number of inner iterations
          is maxiter
    maxiter : {None, int}
        - if restrt is None, maxiter is the max number of inner iterations
          and GMRES does not restart
        - if restrt is int, maxiter is the max number of outer iterations,
          and restrt is the max number of inner iterations
    xtype : type
        dtype for the solution, default is automatic type detection
    M : {array, matrix, sparse matrix, LinearOperator}
        n x n, inverted preconditioner, i.e. solve M A x = M b.
    callback : function
        User-supplied function is called after each iteration as
        callback(xk), where xk is the current solution vector
    residuals : list
        residuals contains the preconditioned residual norm history,
        including the initial residual.
    reorth : boolean
        If True, then a check is made whether to re-orthogonalize the Krylov
        space each GMRES iteration

    Returns
    -------
    (xNew, info)
    xNew : an updated guess to the solution of Ax = b
    info : halting status of gmres

            ==  =============================================
            0   successful exit
            >0  convergence to tolerance not achieved,
                return iteration count instead.  This value
                is precisely the order of the Krylov space.
            <0  numerical breakdown, or illegal input
            ==  =============================================

    Notes
    -----
        - The LinearOperator class is in scipy.sparse.linalg.interface.
          Use this class if you prefer to define A or M as a mat-vec routine
          as opposed to explicitly constructing the matrix.  A.psolve(..) is
          still supported as a legacy.
        - For robustness, modified Gram-Schmidt is used to orthogonalize the
          Krylov Space Givens Rotations are used to provide the residual norm
          each iteration

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
    '''
    # Convert inputs to linear system, with error checking
    A, M, x, b, postprocess = make_system(A, M, x0, b)
    dimen = A.shape[0]

    # Ensure that warnings are always reissued from this function
    import warnings
    warnings.filterwarnings('always', module='pyamg\.krylov\._gmres_mgs')

    # Choose type
    if not hasattr(A, 'dtype'):
        Atype = upcast(x.dtype, b.dtype)
    else:
        Atype = A.dtype
    if not hasattr(M, 'dtype'):
        Mtype = upcast(x.dtype, b.dtype)
    else:
        Mtype = M.dtype
    xtype = upcast(Atype, x.dtype, b.dtype, Mtype)

    if restrt is not None:
        restrt = int(restrt)
    if maxiter is not None:
        maxiter = int(maxiter)

    # Get fast access to underlying BLAS routines
    # dotc is the conjugate dot, dotu does no conjugation
    [lartg] = get_lapack_funcs(['lartg'], [x])
    if np.iscomplexobj(np.zeros((1,), dtype=xtype)):
        [axpy, dotu, dotc, scal] =\
            get_blas_funcs(['axpy', 'dotu', 'dotc', 'scal'], [x])
    else:
        # real type
        [axpy, dotu, dotc, scal] =\
            get_blas_funcs(['axpy', 'dot', 'dot', 'scal'], [x])

    # Make full use of direct access to BLAS by defining own norm
    def norm(z):
        return np.sqrt(np.real(dotc(z, z)))

    # Should norm(r) be kept
    if residuals == []:
        keep_r = True
    else:
        keep_r = False

    # Set number of outer and inner iterations
    if restrt:
        if maxiter:
            max_outer = maxiter
        else:
            max_outer = 1
        if restrt > dimen:
            warn('Setting number of inner iterations (restrt) to maximum\
                  allowed, which is A.shape[0] ')
            restrt = dimen
        max_inner = restrt
    else:
        max_outer = 1
        if maxiter > dimen:
            warn('Setting number of inner iterations (maxiter) to maximum\
                  allowed, which is A.shape[0] ')
            maxiter = dimen
        elif maxiter is None:
            maxiter = min(dimen, 40)
        max_inner = maxiter

    # Is this a one dimensional matrix?
    if dimen == 1:
        entry = np.ravel(A*np.array([1.0], dtype=xtype))
        return (postprocess(b/entry), 0)

    # Prep for method
    r = b - np.ravel(A*x)

    # Apply preconditioner
    r = np.ravel(M*r)
    normr = norm(r)
    if keep_r:
        residuals.append(normr)
    # Check for nan, inf
    # if isnan(r).any() or isinf(r).any():
    #    warn('inf or nan after application of preconditioner')
    #    return(postprocess(x), -1)

    # Check initial guess ( scaling by b, if b != 0,
    #   must account for case when norm(b) is very small)
    normb = norm(b)
    if normb == 0.0:
        normb = 1.0
    if normr < tol*normb:
        return (postprocess(x), 0)

    # Scale tol by ||r_0||_2, we use the preconditioned residual
    # because this is left preconditioned GMRES.
    if normr != 0.0:
        tol = tol*normr

    # Use separate variable to track iterations.  If convergence fails, we
    # cannot simply report niter = (outer-1)*max_outer + inner.  Numerical
    # error could cause the inner loop to halt while the actual ||r|| > tol.
    niter = 0

    # Begin GMRES
    for outer in range(max_outer):

        # Preallocate for Givens Rotations, Hessenberg matrix and Krylov Space
        # Space required is O(dimen*max_inner).
        # NOTE:  We are dealing with row-major matrices, so we traverse in a
        #        row-major fashion,
        #        i.e., H and V's transpose is what we store.
        Q = []  # Givens Rotations
        # Upper Hessenberg matrix, which is then
        #   converted to upper tri with Givens Rots
        H = np.zeros((max_inner+1, max_inner+1), dtype=xtype)
        V = np.zeros((max_inner+1, dimen), dtype=xtype)  # Krylov Space
        # vs store the pointers to each column of V.
        #   This saves a considerable amount of time.
        vs = []
        # v = r/normr
        V[0, :] = scal(1.0/normr, r)
        vs.append(V[0, :])

        # This is the RHS vector for the problem in the Krylov Space
        g = np.zeros((dimen,), dtype=xtype)
        g[0] = normr

        for inner in range(max_inner):

            # New Search Direction
            v = V[inner+1, :]
            v[:] = np.ravel(M*(A*vs[-1]))
            vs.append(v)
            normv_old = norm(v)

            # Check for nan, inf
            # if isnan(V[inner+1, :]).any() or isinf(V[inner+1, :]).any():
            #    warn('inf or nan after application of preconditioner')
            #    return(postprocess(x), -1)

            #  Modified Gram Schmidt
            for k in range(inner+1):
                vk = vs[k]
                alpha = dotc(vk, v)
                H[inner, k] = alpha
                v[:] = axpy(vk, v, dimen, -alpha)

            normv = norm(v)
            H[inner, inner+1] = normv

            # Re-orthogonalize
            if (reorth is True) and (normv_old == normv_old + 0.001*normv):
                for k in range(inner+1):
                    vk = vs[k]
                    alpha = dotc(vk, v)
                    H[inner, k] = H[inner, k] + alpha
                    v[:] = axpy(vk, v, dimen, -alpha)

            # Check for breakdown
            if H[inner, inner+1] != 0.0:
                v[:] = scal(1.0/H[inner, inner+1], v)

            # Apply previous Givens rotations to H
            if inner > 0:
                apply_givens(Q, H[inner, :], inner)

            # Calculate and apply next complex-valued Givens Rotation
            # ==> Note that if max_inner = dimen, then this is unnecessary
            # for the last inner
            #     iteration, when inner = dimen-1.
            if inner != dimen-1:
                if H[inner, inner+1] != 0:
                    [c, s, r] = lartg(H[inner, inner], H[inner, inner+1])
                    Qblock = np.array([[c, s], [-np.conjugate(s), c]],
                                      dtype=xtype)
                    Q.append(Qblock)

                    # Apply Givens Rotation to g,
                    #   the RHS for the linear system in the Krylov Subspace.
                    g[inner:inner+2] = np.dot(Qblock, g[inner:inner+2])

                    # Apply effect of Givens Rotation to H
                    H[inner, inner] = dotu(Qblock[0, :],
                                           H[inner, inner:inner+2])
                    H[inner, inner+1] = 0.0

            niter += 1

            # Don't update normr if last inner iteration, because
            # normr is calculated directly after this loop ends.
            if inner < max_inner-1:
                normr = np.abs(g[inner+1])
                if normr < tol:
                    break

                # Allow user access to the iterates
                if callback is not None:
                    callback(x)
                if keep_r:
                    residuals.append(normr)

        # end inner loop, back to outer loop

        # Find best update to x in Krylov Space V.  Solve inner x inner system.
        y = sp.linalg.solve(H[0:inner+1, 0:inner+1].T, g[0:inner+1])
        update = np.ravel(np.mat(V[:inner+1, :]).T*y.reshape(-1, 1))
        x = x + update
        r = b - np.ravel(A*x)

        # Apply preconditioner
        r = np.ravel(M*r)
        normr = norm(r)
        # Check for nan, inf
        # if isnan(r).any() or isinf(r).any():
        #    warn('inf or nan after application of preconditioner')
        #    return(postprocess(x), -1)

        # Allow user access to the iterates
        if callback is not None:
            callback(x)
        if keep_r:
            residuals.append(normr)

        # Has GMRES stagnated?
        indices = (x != 0)
        if indices.any():
            change = np.max(np.abs(update[indices] / x[indices]))
            if change < 1e-12:
                # No change, halt
                return (postprocess(x), -1)

        # test for convergence
        if normr < tol:
            return (postprocess(x), 0)

    # end outer loop

    return (postprocess(x), niter)


if __name__ == '__main__':
    # from numpy import diag
    # A = random((4,4))
    # A = A*A.transpose() + diag([10,10,10,10])
    # b = random((4,1))
    # x0 = random((4,1))
    # %timeit -n 15 (x,flag) = gmres(A,b,x0,tol=1e-8,maxiter=100)

    from pyamg.gallery import poisson
    from numpy.random import random
    from pyamg.util.linalg import norm
    A = poisson((125, 125), dtype=float, format='csr')
    # A.data = A.data + 0.001j*rand(A.data.shape[0])
    b = random((A.shape[0],))
    x0 = random((A.shape[0],))

    import time
    from scipy.sparse.linalg.isolve import gmres as igmres

    print('\n\nTesting GMRES with %d x %d 2D Laplace Matrix' %
          (A.shape[0], A.shape[0]))
    t1 = time.time()
    (x, flag) = gmres_mgs(A, b, x0, tol=1e-8, maxiter=500)
    t2 = time.time()
    print('%s took %0.3f ms' % ('gmres', (t2-t1)*1000.0))
    print('norm = %g' % (norm(b - A*x)))
    print('info flag = %d' % (flag))

    t1 = time.time()
    # DON"T Enforce a maxiter as scipy gmres can't handle it correctly
    (y, flag) = igmres(A, b, x0, tol=1e-8)
    t2 = time.time()
    print('\n%s took %0.3f ms' % ('linalg gmres', (t2-t1)*1000.0))
    print('norm = %g' % (norm(b - A*y)))
    print('info flag = %d' % (flag))
