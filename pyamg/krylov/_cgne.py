import numpy as np
from scipy.sparse import isspmatrix
from scipy.sparse.sputils import upcast
from scipy.sparse.linalg.isolve.utils import make_system
from scipy.sparse.linalg.interface import aslinearoperator
from warnings import warn
from pyamg.util.linalg import norm


__all__ = ['cgne']


def cgne(A, b, x0=None, tol=1e-5, maxiter=None, xtype=None, M=None,
         callback=None, residuals=None):
    '''Conjugate Gradient, Normal Error algorithm

    Applies CG to the normal equations, A.H A x = b. Left preconditioning
    is supported.  Note that unless A is well-conditioned, the use of
    CGNE is inadvisable

    Parameters
    ----------
    A : {array, matrix, sparse matrix, LinearOperator}
        n x n, linear system to solve
    b : {array, matrix}
        right hand side, shape is (n,) or (n,1)
    x0 : {array, matrix}
        initial guess, default is a vector of zeros
    tol : float
        relative convergence tolerance, i.e. tol is scaled by ||r_0||_2
    maxiter : int
        maximum number of allowed iterations
    xtype : type
        dtype for the solution, default is automatic type detection
    M : {array, matrix, sparse matrix, LinearOperator}
        n x n, inverted preconditioner, i.e. solve M A A.H x = M b.
    callback : function
        User-supplied function is called after each iteration as
        callback(xk), where xk is the current solution vector
    residuals : list
        residuals has the residual norm history,
        including the initial residual, appended to it

    Returns
    -------
    (xNew, info)
    xNew : an updated guess to the solution of Ax = b
    info : halting status of cgne

            ==  =======================================
            0   successful exit
            >0  convergence to tolerance not achieved,
                return iteration count instead.
            <0  numerical breakdown, or illegal input
            ==  =======================================

    Notes
    -----
        - The LinearOperator class is in scipy.sparse.linalg.interface.
          Use this class if you prefer to define A or M as a mat-vec routine
          as opposed to explicitly constructing the matrix.  A.psolve(..) is
          still supported as a legacy.

    Examples
    --------
    >>> from pyamg.krylov.cgne import cgne
    >>> from pyamg.util.linalg import norm
    >>> import numpy as np
    >>> from pyamg.gallery import poisson
    >>> A = poisson((10,10))
    >>> b = np.ones((A.shape[0],))
    >>> (x,flag) = cgne(A,b, maxiter=2, tol=1e-8)
    >>> print norm(b - A*x)
    46.1547104367

    References
    ----------
    .. [1] Yousef Saad, "Iterative Methods for Sparse Linear Systems,
       Second Edition", SIAM, pp. 276-7, 2003
       http://www-users.cs.umn.edu/~saad/books.html

    '''

    # Store the conjugate transpose explicitly as it will be used much later on
    if isspmatrix(A):
        AH = A.H
    else:
        # TODO avoid doing this since A may be a different sparse type
        AH = aslinearoperator(np.asmatrix(A).H)

    # Convert inputs to linear system, with error checking
    A, M, x, b, postprocess = make_system(A, M, x0, b)
    dimen = A.shape[0]

    # Ensure that warnings are always reissued from this function
    import warnings
    warnings.filterwarnings('always', module='pyamg\.krylov\._cgne')

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

    # Should norm(r) be kept
    if residuals == []:
        keep_r = True
    else:
        keep_r = False

    # How often should r be recomputed
    recompute_r = 8

    # Check iteration numbers. CGNE suffers from loss of orthogonality quite
    # easily, so we arbitrarily let the method go up to 130% over the
    # theoretically necessary limit of maxiter=dimen
    if maxiter is None:
        maxiter = int(np.ceil(1.3*dimen)) + 2
    elif maxiter < 1:
        raise ValueError('Number of iterations must be positive')
    elif maxiter > (1.3*dimen):
        warn('maximum allowed inner iterations (maxiter) are the 130% times \
              the number of dofs')
        maxiter = int(np.ceil(1.3*dimen)) + 2

    # Prep for method
    r = b - A*x
    normr = norm(r)
    if keep_r:
        residuals.append(normr)

    # Check initial guess ( scaling by b, if b != 0,
    #   must account for case when norm(b) is very small)
    normb = norm(b)
    if normb == 0.0:
        normb = 1.0
    if normr < tol*normb:
        if callback is not None:
            callback(x)
        return (postprocess(x), 0)

    # Scale tol by ||r_0||_2
    if normr != 0.0:
        tol = tol*normr

    # Begin CGNE

    # Apply preconditioner and calculate initial search direction
    z = M*r
    p = AH*z
    old_zr = np.inner(z.conjugate(), r)

    for iter in range(maxiter):

        # alpha = (z_j, r_j) / (p_j, p_j)
        alpha = old_zr / np.inner(p.conjugate(), p)

        # x_{j+1} = x_j + alpha*p_j
        x += alpha*p

        # r_{j+1} = r_j - alpha*w_j,   where w_j = A*p_j
        if np.mod(iter, recompute_r) and iter > 0:
            r -= alpha*(A*p)
        else:
            r = b - A*x

        # z_{j+1} = M*r_{j+1}
        z = M*r

        # beta = (z_{j+1}, r_{j+1}) / (z_j, r_j)
        new_zr = np.inner(z.conjugate(), r)
        beta = new_zr / old_zr
        old_zr = new_zr

        # p_{j+1} = A.H*z_{j+1} + beta*p_j
        p *= beta
        p += AH*z

        # Allow user access to residual
        if callback is not None:
            callback(x)

        # test for convergence
        normr = norm(r)
        if keep_r:
            residuals.append(normr)
        if normr < tol:
            return (postprocess(x), 0)

    # end loop

    return (postprocess(x), iter+1)

# if __name__ == '__main__':
#    # from numpy import diag
#    # A = random((4,4))
#    # A = A*A.transpose() + diag([10,10,10,10])
#    # b = random((4,1))
#    # x0 = random((4,1))
#
#    from pyamg.gallery import stencil_grid
#    from numpy.random import random
#    A = stencil_grid([[0,-1,0],[-1,4,-1],[0,-1,0]],(150,150), \
#                     dtype=float,format='csr')
#    b = random((A.shape[0],))
#    x0 = random((A.shape[0],))
#
#    import time
#    from scipy.sparse.linalg.isolve import cg as icg
#
#    print '\n\nTesting CGNE with %d x %d 2D Laplace Matrix'%
#    (A.shape[0],A.shape[0])
#    t1=time.time()
#    (x,flag) = cgne(A,b,x0,tol=1e-8,maxiter=100)
#    t2=time.time()
#    print '%s took %0.3f ms' % ('cgne', (t2-t1)*1000.0)
#    print 'norm = %g'%(norm(b - A*x))
#    print 'info flag = %d'%(flag)
#
#    t1=time.time()
#    (y,flag) = icg(A,b,x0,tol=1e-8,maxiter=100)
#    t2=time.time()
#    print '\n%s took %0.3f ms' % ('linalg cg', (t2-t1)*1000.0)
#    print 'norm = %g'%(norm(b - A*y))
#    print 'info flag = %d'%(flag)
#
