from __future__ import print_function
import numpy as np
from scipy.sparse.linalg.isolve.utils import make_system
from scipy.sparse.sputils import upcast
from warnings import warn
from pyamg.util.linalg import norm
from pyamg import amg_core
from scipy.linalg import get_lapack_funcs
import scipy as sp

__all__ = ['gmres_householder']


def mysign(x):
    if x == 0.0:
        return 1.0
    else:
        # return the complex "sign"
        return x / np.abs(x)


def gmres_householder(A, b, x0=None, tol=1e-5, restrt=None, maxiter=None,
                      xtype=None, M=None, callback=None, residuals=None):
    '''
    Generalized Minimum Residual Method (GMRES)
        GMRES iteratively refines the initial solution guess to the
        system Ax = b
        Householder reflections are used for orthogonalization

    Parameters
    ----------
    A : {array, matrix, sparse matrix, LinearOperator}
        n x n, linear system to solve
    b : {array, matrix}
        right hand side, shape is (n,) or (n, 1)
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
        - For robustness, Householder reflections are used to orthonormalize
          the Krylov Space
          Givens Rotations are used to provide the residual norm each iteration

    Examples
    --------
    >>> from pyamg.krylov import gmres
    >>> from pyamg.util.linalg import norm
    >>> import numpy as np
    >>> from pyamg.gallery import poisson
    >>> A = poisson((10, 10))
    >>> b = np.ones((A.shape[0],))
    >>> (x, flag) = gmres(A, b, maxiter=2, tol=1e-8, orthog='householder')
    >>> print norm(b - A*x)
    6.5428213057

    References
    ----------
    .. [1] Yousef Saad, "Iterative Methods for Sparse Linear Systems,
       Second Edition", SIAM, pp. 151-172, pp. 272-275, 2003
       http://www-users.cs.umn.edu/~saad/books.html

    '''
    # Convert inputs to linear system, with error checking
    A, M, x, b, postprocess = make_system(A, M, x0, b)
    dimen = A.shape[0]

    # Ensure that warnings are always reissued from this function
    import warnings
    warnings.filterwarnings('always',
                            module='pyamg\.krylov\._gmres_householder')

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
            warn('Setting number of inner iterations (restrt) to maximum \
                  allowed, which is A.shape[0] ')
            restrt = dimen
        max_inner = restrt
    else:
        max_outer = 1
        if maxiter > dimen:
            warn('Setting number of inner iterations (maxiter) to maximum \
                  allowed, which is A.shape[0] ')
            maxiter = dimen
        elif maxiter is None:
            maxiter = min(dimen, 40)
        max_inner = maxiter

    # Get fast access to underlying LAPACK routine
    [lartg] = get_lapack_funcs(['lartg'], [x])

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

        # Calculate vector w, which defines the Householder reflector
        #    Take shortcut in calculating,
        #    w = r + sign(r[1])*||r||_2*e_1
        w = r
        beta = mysign(w[0])*normr
        w[0] = w[0] + beta
        w[:] = w / norm(w)

        # Preallocate for Krylov vectors, Householder reflectors and
        # Hessenberg matrix
        # Space required is O(dimen*max_inner)
        # Givens Rotations
        Q = np.zeros((4*max_inner,), dtype=xtype)
        # upper Hessenberg matrix (made upper tri with Givens Rotations)
        H = np.zeros((max_inner, max_inner), dtype=xtype)
        # Householder reflectors
        W = np.zeros((max_inner+1, dimen), dtype=xtype)
        W[0, :] = w

        # Multiply r with (I - 2*w*w.T), i.e. apply the Householder reflector
        # This is the RHS vector for the problem in the Krylov Space
        g = np.zeros((dimen,), dtype=xtype)
        g[0] = -beta

        for inner in range(max_inner):
            # Calculate Krylov vector in two steps
            # (1) Calculate v = P_j = (I - 2*w*w.T)v, where k = inner
            v = -2.0*np.conjugate(w[inner])*w
            v[inner] = v[inner] + 1.0
            # (2) Calculate the rest, v = P_1*P_2*P_3...P_{j-1}*ej.
            # for j in range(inner-1,-1,-1):
            #    v -= 2.0*dot(conjugate(W[j,:]), v)*W[j,:]
            amg_core.apply_householders(v, np.ravel(W), dimen, inner-1, -1, -1)

            # Calculate new search direction
            v = np.ravel(A*v)

            # Apply preconditioner
            v = np.ravel(M*v)
            # Check for nan, inf
            # if isnan(v).any() or isinf(v).any():
            #    warn('inf or nan after application of preconditioner')
            #    return(postprocess(x), -1)

            # Factor in all Householder orthogonal reflections on new search
            # direction
            # for j in range(inner+1):
            #    v -= 2.0*dot(conjugate(W[j,:]), v)*W[j,:]
            amg_core.apply_householders(v, np.ravel(W), dimen, 0, inner+1, 1)

            # Calculate next Householder reflector, w
            #  w = v[inner+1:] + sign(v[inner+1])*||v[inner+1:]||_2*e_{inner+1)
            #  Note that if max_inner = dimen, then this is unnecessary for the
            #  last inner iteration, when inner = dimen-1.  Here we do not need
            #  to calculate a Householder reflector or Givens rotation because
            #  nnz(v) is already the desired length, i.e. we do not need to
            #  zero anything out.
            if inner != dimen-1:
                if inner < (max_inner-1):
                    w = W[inner+1, :]
                vslice = v[inner+1:]
                alpha = norm(vslice)
                if alpha != 0:
                    alpha = mysign(vslice[0])*alpha
                    # do not need the final reflector for future calculations
                    if inner < (max_inner-1):
                        w[inner+1:] = vslice
                        w[inner+1] += alpha
                        w[:] = w / norm(w)

                    # Apply new reflector to v
                    #  v = v - 2.0*w*(w.T*v)
                    v[inner+1] = -alpha
                    v[inner+2:] = 0.0

            if inner > 0:
                # Apply all previous Givens Rotations to v
                amg_core.apply_givens(Q, v, dimen, inner)

            # Calculate the next Givens rotation, where j = inner Note that if
            # max_inner = dimen, then this is unnecessary for the last inner
            # iteration, when inner = dimen-1.  Here we do not need to
            # calculate a Householder reflector or Givens rotation because
            # nnz(v) is already the desired length, i.e. we do not need to zero
            # anything out.
            if inner != dimen-1:
                if v[inner+1] != 0:
                    [c, s, r] = lartg(v[inner], v[inner+1])
                    Qblock = np.array([[c, s], [-np.conjugate(s), c]],
                                      dtype=xtype)
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

        # Find best update to x in Krylov Space, V.  Solve inner+1 x inner+1
        # system.  Apparently this is the best way to solve a triangular system
        # in the magical world of scipy
        # piv = arange(inner+1)
        # y = lu_solve((H[0:(inner+1), 0:(inner+1)], piv), g[0:(inner+1)],
        #             trans=0)
        y = sp.linalg.solve(H[0:(inner+1), 0:(inner+1)], g[0:(inner+1)])

        # Use Horner like Scheme to map solution, y, back to original space.
        # Note that we do not use the last reflector.
        update = np.zeros(x.shape, dtype=xtype)
        # for j in range(inner,-1,-1):
        #    update[j] += y[j]
        #    # Apply j-th reflector, (I - 2.0*w_j*w_j.T)*upadate
        #    update -= 2.0*dot(conjugate(W[j,:]), update)*W[j,:]
        amg_core.householder_hornerscheme(update, np.ravel(W), np.ravel(y),
                                          dimen, inner, -1, -1)

        x[:] = x + update
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
    # A = random((4, 4))
    # A = A*A.transpose() + diag([10, 10, 10, 10])
    # b = random((4, 1))
    # x0 = random((4, 1))
    # %timeit -n 15 (x, flag) = gmres(A, b, x0, tol=1e-8, maxiter=100)

    from numpy.random import random
    from pyamg.gallery import poisson
    A = poisson((125, 125), dtype=float, format='csr')
    # A.data = A.data + 0.001j*rand(A.data.shape[0])
    b = random((A.shape[0],))
    x0 = random((A.shape[0],))

    import time
    from scipy.sparse.linalg.isolve import gmres as igmres

    print('\n\nTesting GMRES with %d x %d 2D Laplace Matrix' %
          (A.shape[0], A.shape[0]))
    t1 = time.time()
    (x, flag) = gmres_householder(A, b, x0, tol=1e-8, maxiter=500)
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
