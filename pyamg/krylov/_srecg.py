import numpy as np
from scipy.sparse.linalg.isolve.utils import make_system
from pyamg.util.linalg import norm, BCGS, CGS, split_residual
from ._srecg_orthodir import srecg_orthodir 
from ._srecg_orthodir_new import srecg_orthodir_new 
from ._srecg_bcgs import srecg_bcgs 
from warnings import warn


__all__ = ['srecg']


def srecg(A, b, x0=None, t=1, tol=1e-5, maxiter=None, xtype=None, M=None,
       callback=None, residuals=None, orthog='bcgs', **kwargs):
    '''Short Recurrence Enlarged Conjugate Gradient algorithm

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
    # pass along **kwargs
    if orthog == 'orthodir':
        (x, flag) = srecg_orthodir(A, b, x0=x0, t=t, tol=tol, maxiter=maxiter,
                                    xtype=xtype, M=M, callback=callback,
                                    residuals=residuals, **kwargs)
    if orthog == 'orthodir_new':
        (x, flag) = srecg_orthodir_new(A, b, x0=x0, t=t, tol=tol, maxiter=maxiter,
                                    xtype=xtype, M=M, callback=callback,
                                    residuals=residuals, **kwargs)
    elif orthog == 'bcgs':
        (x, flag) = srecg_bcgs(A, b, x0=x0, t=t, tol=tol, maxiter=maxiter,
                                    xtype=xtype, M=M, callback=callback,
                                    residuals=residuals, **kwargs)
    return (x, flag)

# if __name__ == '__main__':
#    # from numpy import diag
#    # A = random((4,4))
#    # A = A*A.transpose() + diag([10,10,10,10])
#    # b = random((4,1))
#    # x0 = random((4,1))
#
#    from pyamg.gallery import stencil_grid
#    from numpy.random import random
#    A = stencil_grid([[0,-1,0],[-1,4,-1],[0,-1,0]],(100,100),
#                     dtype=float,format='csr')
#    b = random((A.shape[0],))
#    x0 = random((A.shape[0],))
#
#    import time
#    from scipy.sparse.linalg.isolve import cg as icg
#
#    print '\n\nTesting SRECG with %d x %d 2D Laplace Matrix' % \
#           (A.shape[0],A.shape[0])
#    t1=time.time()
#    (x,flag) = srecg(A,b,1,x0,tol=1e-8,maxiter=100)
#    t2=time.time()
#    print '%s took %0.3f ms' % ('srecg', (t2-t1)*1000.0)
#    print 'norm = %g'%(norm(b - A*x))
#    print 'info flag = %d'%(flag)
#
#    t1=time.time()
#    (y,flag) = icg(A,b,x0,tol=1e-8,maxiter=100)
#    t2=time.time()
#    print '\n%s took %0.3f ms' % ('linalg cg', (t2-t1)*1000.0)
#    print 'norm = %g'%(norm(b - A*y))
#    print 'info flag = %d'%(flag)
