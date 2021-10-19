import warnings
from warnings import warn
import numpy as np
from scipy.sparse.linalg.isolve.utils import make_system
from pyamg.util.linalg import norm


__all__ = ['cg']


def cg(A, b, x0=None, tol=1e-5, normA=None,
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
        stopping criteria (see normA)
        ||r_k|| < tol * ||b||, 2-norms
    normA : float
        if provided, then the stopping criteria becomes
        ||r_k|| < tol * (normA * ||x_k|| + ||b||), 2-norms
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
    # must account for case when norm(b) is very small)
    normb = norm(b)
    if normb == 0.0 and normA:
        normb = 1.0
    if normA is not None:
        rtol = tol * (normA * np.linalg.norm(x) + normb)
    else:
        rtol = tol * normb
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

        if normA is not None:
            rtol = tol * (normA * np.linalg.norm(x) + normb)
        else:
            rtol = tol * normb

        if normr < rtol:
            return (postprocess(x), 0)

        if it == maxiter:
            return (postprocess(x), it)


if __name__ == '__main__':
    from pyamg.gallery import stencil_grid
    from numpy.random import random
    import time
    from scipy.sparse.linalg.isolve import cg as icg
    import numpy as np

    nx = 1000
    ny = nx
    A = stencil_grid([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], (nx, ny), dtype=float, format='csr')
    #b = random((A.shape[0],))
    xstar = random((A.shape[0],))
    b = A @ xstar
    x0 = random((A.shape[0],))
    print('initial residual: ', norm(b - A @ x0))
    print('initial criteria 1: ', norm(b - A @ x0) / ((norm(A.data)*norm(x0) + norm(b))))
    print('initial criteria 2: ', norm(b - A @ x0) / norm(b))

    print(f'\n\nTesting CG with {A.shape[0]} x {A.shape[0]} 2D Laplace Matrix')
    x = x0.copy()
    t1 = time.time()
    res = []
    (x, flag) = cg(A, b, x, tol=1e-8, normA=np.linalg.norm(A.data), maxiter=100, residuals=res)
    t2 = time.time()
    #print('res1: ', res)
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
    #print('res2: ', res)
    print(f'\nscipy cg took {(t2-t1)*1000.0} ms')
    print(f'norm = {norm(b - A @ y)}')
    print(f'info flag = {flag}')

    print('-------------')
    norm = np.linalg.norm
    criterion1 = []
    criterion2 = []
    criterion5 = []
    error = []

    def mycb(xk):
        criterion1.append(norm(b - A @ xk) / (norm(A.data)*norm(xk) + norm(b)))
        criterion2.append(norm(b - A @ xk) / norm(b))
        criterion5.append(norm(b - A @ xk) / norm(b - A @ x0))
        error.append(norm(xstar - xk))

    x = x0.copy()
    t1 = time.time()
    res = []
    (x, flag) = cg(A, b, x, tol=1e-8, maxiter=100, callback=mycb)
    t2 = time.time()

    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams['text.usetex'] = True
    plt.semilogy(criterion1, label=r'$\frac{\|r_k\|}{\|b\| + \|A\|\|x_k\|}$')
    plt.semilogy(criterion2, label=r'$\frac{\|r_k\|}{\|b\|}$')
    plt.semilogy(criterion5, label=r'$\frac{\|r_k\|}{\|r_0\|}$')
    plt.semilogy(error, label=r'$\|e_k\|$')
    plt.xlabel('iterations')
    plt.legend()
    plt.savefig('cg.png')
    plt.show()
