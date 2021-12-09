import warnings
from warnings import warn
import numpy as np
from scipy.sparse.linalg.isolve.utils import make_system
import scipy.sparse as sparse
from pyamg.util.linalg import norm


def cg(A, b, x0=None, tol=1e-5, criteria='rr',
       maxiter=None, M=None,
       callback=None, residuals=None):
    """Conjugate Gradient algorithm.

    Solves the linear system Ax = b. Left preconditioning is supported.

    Parameters
    ----------
    A : array, matrix, sparse matrix, LinearOperator
        n x n, linear system to solve
    b : array, matrix
        right hand side, shape is (n,) or (n,1)
    x0 : array, matrix
        initial guess, default is a vector of zeros
    tol : float
        Tolerance for stopping criteria
    criteria : string
        Stopping criteria, let r=r_k, x=x_k
        'rr':        ||r||       < tol ||b||
        'rr+':       ||r||       < tol (||b|| + ||A||_F ||x||)
        'MrMr':      ||M r||     < tol ||M b||
        'rMr':       <r, Mr>^1/2 < tol
        if ||b||=0, then set ||b||=1 for these tests.
    maxiter : int
        maximum number of iterations allowed
    M : array, matrix, sparse matrix, LinearOperator
        n x n, inverse preconditioner, i.e. solve M A x = M b.
    callback : function
        User-supplied function is called after each iteration as
        callback(xk), where xk is the current solution vector
    residuals : list
        residual history in the 2-norm, including the initial residual

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

    Examples
    --------
    >>> from pyamg.krylov.cg import cg
    >>> from pyamg.util.linalg import norm
    >>> import numpy as np
    >>> from pyamg.gallery import poisson
    >>> A = poisson((10,10))
    >>> b = np.ones((A.shape[0],))
    >>> (x,flag) = cg(A,b, maxiter=2, tol=1e-8)
    >>> print norm(b - A @ x)
    10.9370700187

    References
    ----------
    .. [1] Yousef Saad, "Iterative Methods for Sparse Linear Systems,
       Second Edition", SIAM, pp. 262-67, 2003
       http://www-users.cs.umn.edu/~saad/books.html

    """
    # Convert inputs to linear system, with error checking
    A, M, x, b, postprocess = make_system(A, M, x0, b)

    # Ensure that warnings are always reissued from this function
    warnings.filterwarnings('always', module='pyamg.krylov._cg')

    # determine maxiter
    if maxiter is None:
        maxiter = int(1.3*len(b)) + 2
    elif maxiter < 1:
        raise ValueError('Number of iterations must be positive')

    # setup method
    r = b - A @ x
    z = M @ r
    p = z.copy()
    rz = np.inner(r.conjugate(), z)

    normr = np.linalg.norm(r)
    if residuals is not None:
        residuals[:] = [normr]  # initial residual

    # Check initial guess if b != 0,
    normb = norm(b)
    if normb == 0.0:
        normb = 1.0  # reset so that tol is unscaled

    # set the stopping criteria (see the docstring)
    if criteria == 'rr':
        rtol = tol * normb
    elif criteria == 'rr+':
        if sparse.issparse(A.A):
            normA = norm(A.A.data)
        elif isinstance(A.A, np.ndarray):
            normA = norm(np.ravel(A.A))
        else:
            raise ValueError('Unable to use ||A||_F with the current matrix format.')
        rtol = tol * (normA * np.linalg.norm(x) + normb)
    elif criteria == 'MrMr':
        normr = norm(z)
        normMb = norm(M @ b)
        rtol = tol * normMb
    elif criteria == 'rMr':
        normr = np.sqrt(rz)
        rtol = tol
    else:
        raise ValueError('Invalid stopping criteria.')

    if normr < rtol:
        return (postprocess(x), 0)

    # How often should r be recomputed
    recompute_r = 8

    it = 0

    while True:                                   # Step number in Saad's pseudocode
        Ap = A @ p

        rz_old = rz
        pAp = np.inner(Ap.conjugate(), p)         # check curvature of A
        if pAp < 0.0:
            warn("\nIndefinite matrix detected in CG, aborting\n")
            return (postprocess(x), -1)

        alpha = rz/pAp                            # 3
        x += alpha * p                            # 4

        if np.mod(it, recompute_r) and it > 0:    # 5
            r -= alpha * Ap
        else:
            r = b - A @ x

        z = M @ r                                 # 6
        rz = np.inner(r.conjugate(), z)

        if rz < 0.0:                             # check curvature of M
            warn("\nIndefinite preconditioner detected in CG, aborting\n")
            return (postprocess(x), -1)

        beta = rz/rz_old                          # 7
        p *= beta                                 # 8
        p += z

        it += 1

        normr = np.linalg.norm(r)

        if residuals is not None:
            residuals.append(normr)

        if callback is not None:
            callback(x)

        # set the stopping criteria (see the docstring)
        if criteria == 'rr':
            rtol = tol * normb
        elif criteria == 'rr+':
            rtol = tol * (normA * np.linalg.norm(x) + normb)
        elif criteria == 'MrMr':
            normr = norm(z)
            rtol = tol * normMb
        elif criteria == 'rMr':
            normr = np.sqrt(rz)
            rtol = tol

        if normr < rtol:
            return (postprocess(x), 0)

        if it == maxiter:
            return (postprocess(x), it)


if __name__ == '__main__':
    from pyamg.gallery import stencil_grid
    from numpy.random import random
    import time
    from scipy.sparse.linalg.isolve import cg as icg

    nx = 100
    ny = nx
    A = stencil_grid([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], (nx, ny),
                     dtype=float, format='csr')
    # b = random((A.shape[0],))
    xstar = random((A.shape[0],))
    b = A @ xstar
    x0 = random((A.shape[0],))
    print('initial residual: ', norm(b - A @ x0))
    print('initial criteria 1: ', norm(b - A @ x0) / (norm(A.data)*norm(x0) + norm(b)))
    print('initial criteria 2: ', norm(b - A @ x0) / norm(b))

    print(f'\n\nTesting CG with {A.shape[0]} x {A.shape[0]} 2D Laplace Matrix')
    x = x0.copy()
    t1 = time.time()
    res = []
    (x, flag) = cg(A, b, x, tol=1e-8, criteria='rr+', maxiter=100, residuals=res)
    t2 = time.time()
    # print('res1: ', res)
    print(f'cg took {(t2-t1)*1000.0} ms')
    print(f'norm = {norm(b - A @ x)}')
    print(f'info flag = {flag}')

    res = []

    def mycb(xk):
        res.append(norm(b - A @ xk))

    x = x0.copy()
    t1 = time.time()
    (y, flag) = icg(A, b, x, tol=1e-8, maxiter=100, callback=mycb)
    t2 = time.time()
    # print('res2: ', res)
    print(f'\nscipy cg took {(t2-t1)*1000.0} ms')
    print(f'norm = {norm(b - A @ y)}')
    print(f'info flag = {flag}')

    print('-------------')
    norm = np.linalg.norm
    criterion1 = []
    criterion2 = []
    criterion5 = []
    error = []
    errorA = []
    rz = []
    zz = []

    def mycb2(xk, M):
        r = b - A @ xk
        z = M @ r
        e = xstar - xk
        normr = norm(r)
        norme = norm(e)
        criterion1.append(normr / (norm(A.data)*norm(xk) + norm(b)))
        criterion2.append(normr / norm(b))
        criterion5.append(normr / norm(b - A @ x0))
        error.append(norme)
        rz.append(np.sqrt(np.inner(r, z)))
        zz.append(norm(z) / norm(M @ b))
        errorA.append(np.sqrt(np.inner(A @ e, e)))

    import pyamg
    res = []
    ml = pyamg.smoothed_aggregation_solver(A, max_coarse=10, smooth=None)
    M = ml.aspreconditioner()
    x = x0.copy()
    t1 = time.time()
    res = []
    (x, flag) = cg(A, b, x, M=M, tol=1e-16, maxiter=100, callback=mycb2, residuals=res)
    t2 = time.time()

    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams['text.usetex'] = True
    plt.semilogy(criterion1, label=r'$\frac{\|r_k\|}{\|b\| + \|A\|\|x_k\|}$')
    plt.semilogy(criterion2, label=r'$\frac{\|r_k\|}{\|b\|}$')
    plt.semilogy(criterion5, label=r'$\frac{\|r_k\|}{\|r_0\|}$')
    plt.semilogy(error, label=r'$\|e_k\|$')
    plt.semilogy(errorA, label=r'$\|e_k\|_A$')
    plt.semilogy(rz, label=r'$\sqrt{<r, z>}$')
    plt.semilogy(zz, label=r'$\|Mr\| / \|M b\|$')
    plt.hlines(1e-8, 1, len(res), color='tab:gray', linestyle='dashed')
    plt.xlabel('iterations')
    plt.grid(True)
    plt.legend()
    plt.savefig('cg.png', dpi=300)
    plt.show()
