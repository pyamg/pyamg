"""Relaxation methods for linear systems"""


from warnings import warn

import numpy as np
import scipy as sp
from scipy import sparse

from pyamg.util.utils import type_prep, get_diagonal, get_block_diag
from pyamg import amg_core
from scipy.linalg import lapack as la

__all__ = ['sor', 'gauss_seidel', 'jacobi', 'polynomial']
__all__ += ['schwarz', 'schwarz_parameters']
__all__ += ['jacobi_ne', 'gauss_seidel_ne', 'gauss_seidel_nr']
__all__ += ['gauss_seidel_indexed', 'block_jacobi', 'block_gauss_seidel']


def make_system(A, x, b, formats=None):
    """
    Return A,x,b suitable for relaxation or raise an exception

    Parameters
    ----------
    A : {sparse-matrix}
        n x n system
    x : {array}
        n-vector, initial guess
    b : {array}
        n-vector, right-hand side
    formats: {'csr', 'csc', 'bsr', 'lil', 'dok',...}
        desired sparse matrix format
        default is no change to A's format

    Returns
    -------
    (A,x,b), where A is in the desired sparse-matrix format
    and x and b are "raveled", i.e. (n,) vectors.

    Notes
    -----
    Does some rudimentary error checking on the system,
    such as checking for compatible dimensions and checking
    for compatible type, i.e. float or complex.

    Examples
    --------
    >>> from pyamg.relaxation.relaxation import make_system
    >>> from pyamg.gallery import poisson
    >>> import numpy as np
    >>> A = poisson((10,10), format='csr')
    >>> x = np.zeros((A.shape[0],1))
    >>> b = np.ones((A.shape[0],1))
    >>> (A,x,b) = make_system(A,x,b,formats=['csc'])
    >>> print str(x.shape)
    (100,)
    >>> print str(b.shape)
    (100,)
    >>> print A.format
    csc
    """

    if formats is None:
        pass
    elif formats == ['csr']:
        if sparse.isspmatrix_csr(A):
            pass
        elif sparse.isspmatrix_bsr(A):
            A = A.tocsr()
        else:
            warn('implicit conversion to CSR', sparse.SparseEfficiencyWarning)
            A = sparse.csr_matrix(A)
    else:
        if sparse.isspmatrix(A) and A.format in formats:
            pass
        else:
            A = sparse.csr_matrix(A).asformat(formats[0])

    if not isinstance(x, np.ndarray):
        raise ValueError('expected numpy array for argument x')
    if not isinstance(b, np.ndarray):
        raise ValueError('expected numpy array for argument b')

    M, N = A.shape

    if M != N:
        raise ValueError('expected square matrix')

    if x.shape not in [(M,), (M, 1)]:
        raise ValueError('x has invalid dimensions')
    if b.shape not in [(M,), (M, 1)]:
        raise ValueError('b has invalid dimensions')

    if A.dtype != x.dtype or A.dtype != b.dtype:
        raise TypeError('arguments A, x, and b must have the same dtype')

    if not x.flags.carray:
        raise ValueError('x must be contiguous in memory')

    x = np.ravel(x)
    b = np.ravel(b)

    return A, x, b


def sor(A, x, b, omega, iterations=1, sweep='forward'):
    """Perform SOR iteration on the linear system Ax=b

    Parameters
    ----------
    A : {csr_matrix, bsr_matrix}
        Sparse NxN matrix
    x : ndarray
        Approximate solution (length N)
    b : ndarray
        Right-hand side (length N)
    omega : scalar
        Damping parameter
    iterations : int
        Number of iterations to perform
    sweep : {'forward','backward','symmetric'}
        Direction of sweep

    Returns
    -------
    Nothing, x will be modified in place.

    Notes
    -----
    When omega=1.0, SOR is equivalent to Gauss-Seidel.

    Examples
    --------
    >>> # Use SOR as stand-along solver
    >>> from pyamg.relaxation.relaxation import sor
    >>> from pyamg.gallery import poisson
    >>> from pyamg.util.linalg import norm
    >>> import numpy as np
    >>> A = poisson((10,10), format='csr')
    >>> x0 = np.zeros((A.shape[0],1))
    >>> b = np.ones((A.shape[0],1))
    >>> sor(A, x0, b, 1.33, iterations=10)
    >>> print norm(b-A*x0)
    3.03888724811
    >>> #
    >>> # Use SOR as the multigrid smoother
    >>> from pyamg import smoothed_aggregation_solver
    >>> sa = smoothed_aggregation_solver(A, B=np.ones((A.shape[0],1)),
    ...         coarse_solver='pinv2', max_coarse=50,
    ...         presmoother=('sor', {'sweep':'symmetric', 'omega' : 1.33}),
    ...         postsmoother=('sor', {'sweep':'symmetric', 'omega' : 1.33}))
    >>> x0 = np.zeros((A.shape[0],1))
    >>> residuals=[]
    >>> x = sa.solve(b, x0=x0, tol=1e-8, residuals=residuals)
    """
    A, x, b = make_system(A, x, b, formats=['csr', 'bsr'])

    x_old = np.empty_like(x)

    for i in range(iterations):
        x_old[:] = x

        gauss_seidel(A, x, b, iterations=1, sweep=sweep)

        x *= omega
        x_old *= (1-omega)
        x += x_old


def schwarz(A, x, b, iterations=1, subdomain=None, subdomain_ptr=None,
            inv_subblock=None, inv_subblock_ptr=None, sweep='forward'):
    """Perform Overlapping multiplicative Schwarz on
       the linear system Ax=b

    Parameters
    ----------
    A : {csr_matrix, bsr_matrix}
        Sparse NxN matrix
    x : ndarray
        Approximate solution (length N)
    b : ndarray
        Right-hand side (length N)
    iterations : int
        Number of iterations to perform
    subdomain : {int array}
        Linear array containing each subdomain's elements
    subdomain_ptr : {int array}
        Pointer in subdomain, such that
        subdomain[subdomain_ptr[i]:subdomain_ptr[i+1]]]
        contains the _sorted_ indices in subdomain i
    inv_subblock : {int_array}
        Linear array containing each subdomain's
        inverted diagonal block of A
    inv_subblock_ptr : {int array}
        Pointer in inv_subblock, such that
        inv_subblock[inv_subblock_ptr[i]:inv_subblock_ptr[i+1]]]
        contains the inverted diagonal block of A for the
        i-th subdomain in _row_ major order
    sweep : {'forward','backward','symmetric'}
        Direction of sweep

    Returns
    -------
    Nothing, x will be modified in place.

    Notes
    -----
    If subdomains is None, then a point-wise iteration takes place,
    with the overlapping region defined by each degree-of-freedom's
    neighbors in the matrix graph.

    If subdomains is not None, but subblocks is, then the subblocks
    are formed internally.

    Currently only supports CSR matrices

    Examples
    --------
    >>> # Use Overlapping Schwarz as a Stand-Alone Solver
    >>> from pyamg.relaxation.relaxation import schwarz
    >>> from pyamg.gallery import poisson
    >>> from pyamg.util.linalg import norm
    >>> import numpy as np
    >>> A = poisson((10,10), format='csr')
    >>> x0 = np.zeros((A.shape[0],1))
    >>> b = np.ones((A.shape[0],1))
    >>> schwarz(A, x0, b, iterations=10)
    >>> print norm(b-A*x0)
    0.126326160522
    >>> #
    >>> # Schwarz as the Multigrid Smoother
    >>> from pyamg import smoothed_aggregation_solver
    >>> sa = smoothed_aggregation_solver(A, B=np.ones((A.shape[0],1)),
    ...         coarse_solver='pinv2', max_coarse=50,
    ...         presmoother='schwarz',
    ...         postsmoother='schwarz')
    >>> x0=np.zeros((A.shape[0],1))
    >>> residuals=[]
    >>> x = sa.solve(b, x0=x0, tol=1e-8, residuals=residuals)
    """

    A, x, b = make_system(A, x, b, formats=['csr'])
    A.sort_indices()

    if subdomain is None and inv_subblock is not None:
        raise ValueError("inv_subblock must be None if subdomain is None")

    # If no subdomains are defined, default is to use the sparsity pattern of A
    # to define the overlapping regions
    (subdomain, subdomain_ptr, inv_subblock, inv_subblock_ptr) = \
        schwarz_parameters(A, subdomain, subdomain_ptr,
                           inv_subblock, inv_subblock_ptr)

    if sweep == 'forward':
        row_start, row_stop, row_step = 0, subdomain_ptr.shape[0]-1, 1
    elif sweep == 'backward':
        row_start, row_stop, row_step = subdomain_ptr.shape[0]-2, -1, -1
    elif sweep == 'symmetric':
        for iter in range(iterations):
            schwarz(A, x, b, iterations=1, subdomain=subdomain,
                    subdomain_ptr=subdomain_ptr, inv_subblock=inv_subblock,
                    inv_subblock_ptr=inv_subblock_ptr, sweep='forward')
            schwarz(A, x, b, iterations=1, subdomain=subdomain,
                    subdomain_ptr=subdomain_ptr, inv_subblock=inv_subblock,
                    inv_subblock_ptr=inv_subblock_ptr, sweep='backward')
        return
    else:
        raise ValueError("valid sweep directions are 'forward',\
                          'backward', and 'symmetric'")

    # Call C code, need to make sure that subdomains are sorted and unique
    for iter in range(iterations):
        amg_core.overlapping_schwarz_csr(A.indptr, A.indices, A.data,
                                         x, b, inv_subblock, inv_subblock_ptr,
                                         subdomain, subdomain_ptr,
                                         subdomain_ptr.shape[0]-1, A.shape[0],
                                         row_start, row_stop, row_step)


def gauss_seidel(A, x, b, iterations=1, sweep='forward'):
    """Perform Gauss-Seidel iteration on the linear system Ax=b

    Parameters
    ----------
    A : {csr_matrix, bsr_matrix}
        Sparse NxN matrix
    x : ndarray
        Approximate solution (length N)
    b : ndarray
        Right-hand side (length N)
    iterations : int
        Number of iterations to perform
    sweep : {'forward','backward','symmetric'}
        Direction of sweep

    Returns
    -------
    Nothing, x will be modified in place.

    Examples
    --------
    >>> # Use Gauss-Seidel as a Stand-Alone Solver
    >>> from pyamg.relaxation.relaxation import gauss_seidel
    >>> from pyamg.gallery import poisson
    >>> from pyamg.util.linalg import norm
    >>> import numpy as np
    >>> A = poisson((10,10), format='csr')
    >>> x0 = np.zeros((A.shape[0],1))
    >>> b = np.ones((A.shape[0],1))
    >>> gauss_seidel(A, x0, b, iterations=10)
    >>> print norm(b-A*x0)
    4.00733716236
    >>> #
    >>> # Use Gauss-Seidel as the Multigrid Smoother
    >>> from pyamg import smoothed_aggregation_solver
    >>> sa = smoothed_aggregation_solver(A, B=np.ones((A.shape[0],1)),
    ...         coarse_solver='pinv2', max_coarse=50,
    ...         presmoother=('gauss_seidel', {'sweep':'symmetric'}),
    ...         postsmoother=('gauss_seidel', {'sweep':'symmetric'}))
    >>> x0=np.zeros((A.shape[0],1))
    >>> residuals=[]
    >>> x = sa.solve(b, x0=x0, tol=1e-8, residuals=residuals)
    """
    A, x, b = make_system(A, x, b, formats=['csr', 'bsr'])

    if sparse.isspmatrix_csr(A):
        blocksize = 1
    else:
        R, C = A.blocksize
        if R != C:
            raise ValueError('BSR blocks must be square')
        blocksize = R

    if sweep == 'forward':
        row_start, row_stop, row_step = 0, int(len(x)/blocksize), 1
    elif sweep == 'backward':
        row_start, row_stop, row_step = int(len(x)/blocksize)-1, -1, -1
    elif sweep == 'symmetric':
        for iter in range(iterations):
            gauss_seidel(A, x, b, iterations=1, sweep='forward')
            gauss_seidel(A, x, b, iterations=1, sweep='backward')
        return
    else:
        raise ValueError("valid sweep directions are 'forward',\
                          'backward', and 'symmetric'")

    if sparse.isspmatrix_csr(A):
        for iter in range(iterations):
            amg_core.gauss_seidel(A.indptr, A.indices, A.data, x, b,
                                  row_start, row_stop, row_step)
    else:
        for iter in range(iterations):
            amg_core.bsr_gauss_seidel(A.indptr, A.indices, np.ravel(A.data),
                                      x, b, row_start, row_stop, row_step, R)


def jacobi(A, x, b, iterations=1, omega=1.0):
    """Perform Jacobi iteration on the linear system Ax=b

    Parameters
    ----------
    A : csr_matrix
        Sparse NxN matrix
    x : ndarray
        Approximate solution (length N)
    b : ndarray
        Right-hand side (length N)
    iterations : int
        Number of iterations to perform
    omega : scalar
        Damping parameter

    Returns
    -------
    Nothing, x will be modified in place.

    Examples
    --------
    >>> # Use Jacobi as a Stand-Alone Solver
    >>> from pyamg.relaxation.relaxation.relaxation import jacobi
    >>> from pyamg.gallery import poisson
    >>> from pyamg.util.linalg import norm
    >>> import numpy as np
    >>> A = poisson((10,10), format='csr')
    >>> x0 = np.zeros((A.shape[0],1))
    >>> b = np.ones((A.shape[0],1))
    >>> jacobi(A, x0, b, iterations=10, omega=1.0)
    >>> print norm(b-A*x0)
    5.83475132751
    >>> #
    >>> # Use Jacobi as the Multigrid Smoother
    >>> from pyamg import smoothed_aggregation_solver
    >>> sa = smoothed_aggregation_solver(A, B=np.ones((A.shape[0],1)),
    ...         coarse_solver='pinv2', max_coarse=50,
    ...         presmoother=('jacobi', {'omega': 4.0/3.0, 'iterations' : 2}),
    ...         postsmoother=('jacobi', {'omega': 4.0/3.0, 'iterations' : 2}))
    >>> x0=np.zeros((A.shape[0],1))
    >>> residuals=[]
    >>> x = sa.solve(b, x0=x0, tol=1e-8, residuals=residuals)
    """
    A, x, b = make_system(A, x, b, formats=['csr', 'bsr'])

    sweep = slice(None)
    (row_start, row_stop, row_step) = sweep.indices(A.shape[0])

    if (row_stop - row_start) * row_step <= 0:  # no work to do
        return

    temp = np.empty_like(x)

    # Create uniform type, convert possibly complex scalars to length 1 arrays
    [omega] = type_prep(A.dtype, [omega])

    if sparse.isspmatrix_csr(A):
        for iter in range(iterations):
            amg_core.jacobi(A.indptr, A.indices, A.data, x, b, temp,
                            row_start, row_stop, row_step, omega)
    else:
        R, C = A.blocksize
        if R != C:
            raise ValueError('BSR blocks must be square')
        row_start = int(row_start / R)
        row_stop = int(row_stop / R)
        for iter in range(iterations):
            amg_core.bsr_jacobi(A.indptr, A.indices, np.ravel(A.data),
                                x, b, temp, row_start, row_stop,
                                row_step, R, omega)


def block_jacobi(A, x, b, Dinv=None, blocksize=1, iterations=1, omega=1.0):
    """Perform block Jacobi iteration on the linear system Ax=b

    Parameters
    ----------
    A : csr_matrix or bsr_matrix
        Sparse NxN matrix
    x : ndarray
        Approximate solution (length N)
    b : ndarray
        Right-hand side (length N)
    Dinv : array
        Array holding block diagonal inverses of A
        size (N/blocksize, blocksize, blocksize)
    blocksize : int
        Desired dimension of blocks
    iterations : int
        Number of iterations to perform
    omega : scalar
        Damping parameter

    Returns
    -------
    Nothing, x will be modified in place.

    Examples
    --------
    >>> # Use block Jacobi as a Stand-Alone Solver
    >>> from pyamg.relaxation.relaxation import block_jacobi
    >>> from pyamg.gallery import poisson
    >>> from pyamg.util.linalg import norm
    >>> import numpy as np
    >>> A = poisson((10,10), format='csr')
    >>> x0 = np.zeros((A.shape[0],1))
    >>> b = np.ones((A.shape[0],1))
    >>> block_jacobi(A, x0, b, blocksize=4, iterations=10, omega=1.0)
    >>> print norm(b-A*x0)
    4.66474230129
    >>> #
    >>> # Use block Jacobi as the Multigrid Smoother
    >>> from pyamg import smoothed_aggregation_solver
    >>> opts = {'omega': 4.0/3.0, 'iterations' : 2, 'blocksize' : 4}
    >>> sa = smoothed_aggregation_solver(A, B=np.ones((A.shape[0],1)),
    ...        coarse_solver='pinv2', max_coarse=50,
    ...        presmoother=('block_jacobi', opts),
    ...        postsmoother=('block_jacobi', opts))
    >>> x0=np.zeros((A.shape[0],1))
    >>> residuals=[]
    >>> x = sa.solve(b, x0=x0, tol=1e-8, residuals=residuals)
    """

    A, x, b = make_system(A, x, b, formats=['csr', 'bsr'])
    A = A.tobsr(blocksize=(blocksize, blocksize))

    if Dinv is None:
        Dinv = get_block_diag(A, blocksize=blocksize, inv_flag=True)
    elif Dinv.shape[0] != int(A.shape[0]/blocksize):
        raise ValueError('Dinv and A have incompatible dimensions')
    elif (Dinv.shape[1] != blocksize) or (Dinv.shape[2] != blocksize):
        raise ValueError('Dinv and blocksize are incompatible')

    sweep = slice(None)
    (row_start, row_stop, row_step) = sweep.indices(int(A.shape[0]/blocksize))

    if (row_stop - row_start) * row_step <= 0:  # no work to do
        return

    temp = np.empty_like(x)

    # Create uniform type, convert possibly complex scalars to length 1 arrays
    [omega] = type_prep(A.dtype, [omega])

    for iter in range(iterations):
        amg_core.block_jacobi(A.indptr, A.indices, np.ravel(A.data),
                              x, b, np.ravel(Dinv), temp,
                              row_start, row_stop, row_step,
                              omega, blocksize)


def block_gauss_seidel(A, x, b, iterations=1, sweep='forward', blocksize=1,
                       Dinv=None):
    """Perform block Gauss-Seidel iteration on the linear system Ax=b

    Parameters
    ----------
    A : {csr_matrix, bsr_matrix}
        Sparse NxN matrix
    x : ndarray
        Approximate solution (length N)
    b : ndarray
        Right-hand side (length N)
    iterations : int
        Number of iterations to perform
    sweep : {'forward','backward','symmetric'}
        Direction of sweep
    Dinv : array
        Array holding block diagonal inverses of A
        size (N/blocksize, blocksize, blocksize)
    blocksize : int
        Desired dimension of blocks


    Returns
    -------
    Nothing, x will be modified in place.

    Examples
    --------
    >>> # Use Gauss-Seidel as a Stand-Alone Solver
    >>> from pyamg.relaxation.relaxation import block_gauss_seidel
    >>> from pyamg.gallery import poisson
    >>> from pyamg.util.linalg import norm
    >>> import numpy as np
    >>> A = poisson((10,10), format='csr')
    >>> x0 = np.zeros((A.shape[0],1))
    >>> b = np.ones((A.shape[0],1))
    >>> block_gauss_seidel(A, x0, b, iterations=10, blocksize=4,
                           sweep='symmetric')
    >>> print norm(b-A*x0)
    0.958333817624
    >>> #
    >>> # Use Gauss-Seidel as the Multigrid Smoother
    >>> from pyamg import smoothed_aggregation_solver
    >>> opts = {'sweep':'symmetric', 'blocksize' : 4}
    >>> sa = smoothed_aggregation_solver(A, B=np.ones((A.shape[0],1)),
    ...        coarse_solver='pinv2', max_coarse=50,
    ...        presmoother=('block_gauss_seidel', opts),
    ...        postsmoother=('block_gauss_seidel', opts))
    >>> x0=np.zeros((A.shape[0],1))
    >>> residuals=[]
    >>> x = sa.solve(b, x0=x0, tol=1e-8, residuals=residuals)
    """
    A, x, b = make_system(A, x, b, formats=['csr', 'bsr'])
    A = A.tobsr(blocksize=(blocksize, blocksize))

    if Dinv is None:
        Dinv = get_block_diag(A, blocksize=blocksize, inv_flag=True)
    elif Dinv.shape[0] != int(A.shape[0]/blocksize):
        raise ValueError('Dinv and A have incompatible dimensions')
    elif (Dinv.shape[1] != blocksize) or (Dinv.shape[2] != blocksize):
        raise ValueError('Dinv and blocksize are incompatible')

    if sweep == 'forward':
        row_start, row_stop, row_step = 0, int(len(x)/blocksize), 1
    elif sweep == 'backward':
        row_start, row_stop, row_step = int(len(x)/blocksize)-1, -1, -1
    elif sweep == 'symmetric':
        for iter in range(iterations):
            block_gauss_seidel(A, x, b, iterations=1, sweep='forward',
                               blocksize=blocksize, Dinv=Dinv)
            block_gauss_seidel(A, x, b, iterations=1, sweep='backward',
                               blocksize=blocksize, Dinv=Dinv)
        return
    else:
        raise ValueError("valid sweep directions are 'forward',\
                          'backward', and 'symmetric'")

    for iter in range(iterations):
        amg_core.block_gauss_seidel(A.indptr, A.indices, np.ravel(A.data),
                                    x, b, np.ravel(Dinv),
                                    row_start, row_stop, row_step, blocksize)


def polynomial(A, x, b, coefficients, iterations=1):
    """Apply a polynomial smoother to the system Ax=b


    Parameters
    ----------
    A : sparse matrix
        Sparse NxN matrix
    x : ndarray
        Approximate solution (length N)
    b : ndarray
        Right-hand side (length N)
    coefficients : {array_like}
        Coefficients of the polynomial.  See Notes section for details.
    iterations : int
        Number of iterations to perform

    Returns
    -------
    Nothing, x will be modified in place.

    Notes
    -----
    The smoother has the form  x[:] = x + p(A) (b - A*x) where p(A) is a
    polynomial in A whose scalar coefficients are specified (in descending
    order) by argument 'coefficients'.

    - Richardson iteration p(A) = c_0:
        polynomial_smoother(A, x, b, [c_0])

    - Linear smoother p(A) = c_1*A + c_0:
        polynomial_smoother(A, x, b, [c_1, c_0])

    - Quadratic smoother p(A) = c_2*A^2 + c_1*A + c_0:
        polynomial_smoother(A, x, b, [c_2, c_1, c_0])

    Here, Horner's Rule is applied to avoid computing A^k directly.

    For efficience, the method detects the case x = 0 one matrix-vector
    product is avoided (since (b - A*x) is b).

    Examples
    --------
    >>> # The polynomial smoother is not currently used directly
    >>> # in PyAMG.  It is only used by the chebyshev smoothing option,
    >>> # which automatically calculates the correct coefficients.
    >>> from pyamg.gallery import poisson
    >>> from pyamg.util.linalg import norm
    >>> import numpy as np
    >>> from pyamg.aggregation import smoothed_aggregation_solver
    >>> A = poisson((10,10), format='csr')
    >>> b = np.ones((A.shape[0],1))
    >>> sa = smoothed_aggregation_solver(A, B=np.ones((A.shape[0],1)),
    ...         coarse_solver='pinv2', max_coarse=50,
    ...         presmoother=('chebyshev', {'degree':3, 'iterations':1}),
    ...         postsmoother=('chebyshev', {'degree':3, 'iterations':1}))
    >>> x0=np.zeros((A.shape[0],1))
    >>> residuals=[]
    >>> x = sa.solve(b, x0=x0, tol=1e-8, residuals=residuals)
    """
    A, x, b = make_system(A, x, b, formats=None)

    for i in range(iterations):
        from pyamg.util.linalg import norm

        if norm(x) == 0:
            residual = b
        else:
            residual = (b - A*x)

        h = coefficients[0]*residual

        for c in coefficients[1:]:
            h = c*residual + A*h

        x += h


def gauss_seidel_indexed(A, x, b, indices, iterations=1, sweep='forward'):
    """Perform indexed Gauss-Seidel iteration on the linear system Ax=b

    In indexed Gauss-Seidel, the sequence in which unknowns are relaxed is
    specified explicitly.  In contrast, the standard Gauss-Seidel method
    always performs complete sweeps of all variables in increasing or
    decreasing order.  The indexed method may be used to implement
    specialized smoothers, like F-smoothing in Classical AMG.

    Parameters
    ----------
    A : csr_matrix
        Sparse NxN matrix
    x : ndarray
        Approximate solution (length N)
    b : ndarray
        Right-hand side (length N)
    indices : ndarray
        Row indices to relax.
    iterations : int
        Number of iterations to perform
    sweep : {'forward','backward','symmetric'}
        Direction of sweep

    Returns
    -------
    Nothing, x will be modified in place.

    Examples
    --------
    >>> from pyamg.gallery import poisson
    >>> from pyamg.relaxation.relaxation import gauss_seidel_indexed
    >>> import numpy as np
    >>> A = poisson((4,), format='csr')
    >>> x = np.array([0.0, 0.0, 0.0, 0.0])
    >>> b = np.array([0.0, 1.0, 2.0, 3.0])
    >>> gauss_seidel_indexed(A, x, b, [0,1,2,3])  # relax all rows in order
    >>> gauss_seidel_indexed(A, x, b, [0,1])      # relax first two rows
    >>> gauss_seidel_indexed(A, x, b, [2,0])      # relax row 2, then row 0
    >>> gauss_seidel_indexed(A, x, b, [2,3], sweep='backward')  # 3, then 2
    >>> gauss_seidel_indexed(A, x, b, [2,0,2])    # relax row 2, 0, 2

    """
    A, x, b = make_system(A, x, b, formats=['csr'])

    indices = np.asarray(indices, dtype='intc')

    # if indices.min() < 0:
    #     raise ValueError('row index (%d) is invalid' % indices.min())
    # if indices.max() >= A.shape[0]
    #     raise ValueError('row index (%d) is invalid' % indices.max())

    if sweep == 'forward':
        row_start, row_stop, row_step = 0, len(indices), 1
    elif sweep == 'backward':
        row_start, row_stop, row_step = len(indices)-1, -1, -1
    elif sweep == 'symmetric':
        for iter in range(iterations):
            gauss_seidel_indexed(A, x, b, indices, iterations=1,
                                 sweep='forward')
            gauss_seidel_indexed(A, x, b, indices, iterations=1,
                                 sweep='backward')
        return
    else:
        raise ValueError("valid sweep directions are 'forward',\
                          'backward', and 'symmetric'")

    for iter in range(iterations):
        amg_core.gauss_seidel_indexed(A.indptr, A.indices, A.data,
                                      x, b, indices,
                                      row_start, row_stop, row_step)


def jacobi_ne(A, x, b, iterations=1, omega=1.0):
    """Perform Jacobi iterations on the linear system A A.H x = A.H b
       (Also known as Cimmino relaxation)

    Parameters
    ----------
    A : csr_matrix
        Sparse NxN matrix
    x : ndarray
        Approximate solution (length N)
    b : ndarray
        Right-hand side (length N)
    iterations : int
        Number of iterations to perform
    omega : scalar
        Damping parameter

    Returns
    -------
    Nothing, x will be modified in place.

    References
    ----------
    .. [1] Brandt, Ta'asan.
       "Multigrid Method For Nearly Singular And Slightly Indefinite Problems."
       1985.  NASA Technical Report Numbers: ICASE-85-57; NAS 1.26:178026;
       NASA-CR-178026;

    .. [2] Kaczmarz.  Angenaeherte Aufloesung von Systemen Linearer
       Gleichungen.  Bull. Acad.  Polon. Sci. Lett. A 35, 355-57.  1937

    .. [3] Cimmino. La ricerca scientifica ser. II 1.
       Pubbliz. dell'Inst. pre le Appl. del Calculo 34, 326-333, 1938.

    Examples
    --------
    >>> # Use NE Jacobi as a Stand-Alone Solver
    >>> from pyamg.relaxation.relaxation import jacobi_ne
    >>> from pyamg.gallery import poisson
    >>> from pyamg.util.linalg import norm
    >>> import numpy as np
    >>> A = poisson((50,50), format='csr')
    >>> x0 = np.zeros((A.shape[0],1))
    >>> b = np.ones((A.shape[0],1))
    >>> jacobi_ne(A, x0, b, iterations=10, omega=2.0/3.0)
    >>> print norm(b-A*x0)
    49.3886046066
    >>> #
    >>> # Use NE Jacobi as the Multigrid Smoother
    >>> from pyamg import smoothed_aggregation_solver
    >>> opts = {'iterations' : 2, 'omega' : 4.0/3.0}
    >>> sa = smoothed_aggregation_solver(A, B=np.ones((A.shape[0],1)),
    ...         coarse_solver='pinv2', max_coarse=50,
    ...         presmoother=('jacobi_ne', opts),
    ...         postsmoother=('jacobi_ne', opts))
    >>> x0=np.zeros((A.shape[0],1))
    >>> residuals=[]
    >>> x = sa.solve(b, x0=x0, tol=1e-8, residuals=residuals)
    """
    A, x, b = make_system(A, x, b, formats=['csr'])

    sweep = slice(None)
    (row_start, row_stop, row_step) = sweep.indices(A.shape[0])

    temp = np.zeros_like(x)

    # Dinv for A*A.H
    Dinv = get_diagonal(A, norm_eq=2, inv=True)

    # Create uniform type, convert possibly complex scalars to length 1 arrays
    [omega] = type_prep(A.dtype, [omega])

    for i in range(iterations):
        delta = (np.ravel(b - A*x)*np.ravel(Dinv)).astype(A.dtype)
        amg_core.jacobi_ne(A.indptr, A.indices, A.data,
                           x, b, delta, temp, row_start,
                           row_stop, row_step, omega)


def gauss_seidel_ne(A, x, b, iterations=1, sweep='forward', omega=1.0,
                    Dinv=None):
    """Perform Gauss-Seidel iterations on the linear system A A.H x = b
       (Also known as Kaczmarz relaxation)

    Parameters
    ----------
    A : csr_matrix
        Sparse NxN matrix
    x : { ndarray }
        Approximate solution (length N)
    b : { ndarray }
        Right-hand side (length N)
    iterations : { int }
        Number of iterations to perform
    sweep : {'forward','backward','symmetric'}
        Direction of sweep
    omega : { float}
        Relaxation parameter typically in (0, 2)
        if omega != 1.0, then algorithm becomes SOR on A A.H
    Dinv : { ndarray}
        Inverse of diag(A A.H),  (length N)

    Returns
    -------
    Nothing, x will be modified in place.

    References
    ----------
    .. [1] Brandt, Ta'asan.
       "Multigrid Method For Nearly Singular And Slightly Indefinite Problems."
       1985.  NASA Technical Report Numbers: ICASE-85-57; NAS 1.26:178026;
       NASA-CR-178026;

    .. [2] Kaczmarz.  Angenaeherte Aufloesung von Systemen Linearer
       Gleichungen. Bull. Acad.  Polon. Sci. Lett. A 35, 355-57.  1937

    Examples
    --------
    >>> # Use NE Gauss-Seidel as a Stand-Alone Solver
    >>> from pyamg.relaxation.relaxation import gauss_seidel_ne
    >>> from pyamg.gallery import poisson
    >>> from pyamg.util.linalg import norm
    >>> import numpy as np
    >>> A = poisson((10,10), format='csr')
    >>> x0 = np.zeros((A.shape[0],1))
    >>> b = np.ones((A.shape[0],1))
    >>> gauss_seidel_ne(A, x0, b, iterations=10, sweep='symmetric')
    >>> print norm(b-A*x0)
    8.47576806771
    >>> #
    >>> # Use NE Gauss-Seidel as the Multigrid Smoother
    >>> from pyamg import smoothed_aggregation_solver
    >>> sa = smoothed_aggregation_solver(A, B=np.ones((A.shape[0],1)),
    ...         coarse_solver='pinv2', max_coarse=50,
    ...         presmoother=('gauss_seidel_ne', {'sweep' : 'symmetric'}),
    ...         postsmoother=('gauss_seidel_ne', {'sweep' : 'symmetric'}))
    >>> x0=np.zeros((A.shape[0],1))
    >>> residuals=[]
    >>> x = sa.solve(b, x0=x0, tol=1e-8, residuals=residuals)
    """

    A, x, b = make_system(A, x, b, formats=['csr'])

    # Dinv for A*A.H
    if Dinv is None:
        Dinv = np.ravel(get_diagonal(A, norm_eq=2, inv=True))

    if sweep == 'forward':
        row_start, row_stop, row_step = 0, len(x), 1
    elif sweep == 'backward':
        row_start, row_stop, row_step = len(x)-1, -1, -1
    elif sweep == 'symmetric':
        for iter in range(iterations):
            gauss_seidel_ne(A, x, b, iterations=1, sweep='forward',
                            omega=omega, Dinv=Dinv)
            gauss_seidel_ne(A, x, b, iterations=1, sweep='backward',
                            omega=omega, Dinv=Dinv)
        return
    else:
        raise ValueError("valid sweep directions are 'forward',\
                          'backward', and 'symmetric'")

    for i in range(iterations):
        amg_core.gauss_seidel_ne(A.indptr, A.indices, A.data,
                                 x, b, row_start,
                                 row_stop, row_step, Dinv, omega)


def gauss_seidel_nr(A, x, b, iterations=1, sweep='forward', omega=1.0,
                    Dinv=None):
    """Perform Gauss-Seidel iterations on the linear system A.H A x = A.H b

    Parameters
    ----------
    A : csr_matrix
        Sparse NxN matrix
    x : { ndarray }
        Approximate solution (length N)
    b : { ndarray }
        Right-hand side (length N)
    iterations : { int }
        Number of iterations to perform
    sweep : {'forward','backward','symmetric'}
        Direction of sweep
    omega : { float}
        Relaxation parameter typically in (0, 2)
        if omega != 1.0, then algorithm becomes SOR on A.H A
    Dinv : { ndarray}
        Inverse of diag(A.H A),  (length N)

    Returns
    -------
    Nothing, x will be modified in place.

    References
    ----------
    .. [1] Yousef Saad, "Iterative Methods for Sparse Linear Systems,
       Second Edition", SIAM, pp. 247-9, 2003
       http://www-users.cs.umn.edu/~saad/books.html


    Examples
    --------
    >>> # Use NR Gauss-Seidel as a Stand-Alone Solver
    >>> from pyamg.relaxation.relaxation import gauss_seidel_nr
    >>> from pyamg.gallery import poisson
    >>> from pyamg.util.linalg import norm
    >>> import numpy as np
    >>> A = poisson((10,10), format='csr')
    >>> x0 = np.zeros((A.shape[0],1))
    >>> b = np.ones((A.shape[0],1))
    >>> gauss_seidel_nr(A, x0, b, iterations=10, sweep='symmetric')
    >>> print norm(b-A*x0)
    8.45044864352
    >>> #
    >>> # Use NR Gauss-Seidel as the Multigrid Smoother
    >>> from pyamg import smoothed_aggregation_solver
    >>> sa = smoothed_aggregation_solver(A, B=np.ones((A.shape[0],1)),
    ...      coarse_solver='pinv2', max_coarse=50,
    ...      presmoother=('gauss_seidel_nr', {'sweep' : 'symmetric'}),
    ...      postsmoother=('gauss_seidel_nr', {'sweep' : 'symmetric'}))
    >>> x0=np.zeros((A.shape[0],1))
    >>> residuals=[]
    >>> x = sa.solve(b, x0=x0, tol=1e-8, residuals=residuals)
    """

    A, x, b = make_system(A, x, b, formats=['csc'])

    # Dinv for A.H*A
    if Dinv is None:
        Dinv = np.ravel(get_diagonal(A, norm_eq=1, inv=True))

    if sweep == 'forward':
        col_start, col_stop, col_step = 0, len(x), 1
    elif sweep == 'backward':
        col_start, col_stop, col_step = len(x)-1, -1, -1
    elif sweep == 'symmetric':
        for iter in range(iterations):
            gauss_seidel_nr(A, x, b, iterations=1, sweep='forward',
                            omega=omega, Dinv=Dinv)
            gauss_seidel_nr(A, x, b, iterations=1, sweep='backward',
                            omega=omega, Dinv=Dinv)
        return
    else:
        raise ValueError("valid sweep directions are 'forward',\
                          'backward', and 'symmetric'")

    # Calculate initial residual
    r = b - A*x

    for i in range(iterations):
        amg_core.gauss_seidel_nr(A.indptr, A.indices, A.data,
                                 x, r, col_start,
                                 col_stop, col_step, Dinv, omega)

# Appends
# A.schwarz_tuple = (subdomain, subdomain_ptr, inv_subblock, inv_subblock_ptr)
# to the matrix passed in If this tuple already exists, return it, otherwise
# compute it Should work for passing in both A and the Strength matrix C Make
# sure to wrap an Acsr into the schwarz call
# make sure that it handles passing in preset subdomain stuff, and recomputing
# it

# If subdomain and subdomain_ptr are passed in, check to see that they are the
# same as any preexisting subdomain and subdomain_ptr?


def schwarz_parameters(A, subdomain=None, subdomain_ptr=None,
                       inv_subblock=None, inv_subblock_ptr=None):
    '''
    Helper function for setting up Schwarz relaxation.  This function avoids
    recomputing the subdomains and block inverses manytimes, e.g., it avoids a
    costly double computation when setting up pre and post smoothing with
    Schwarz.

    Parameters
    ----------
    A {csr_matrix}

    Returns
    -------
    A.schwarz_parameters[0] is subdomain
    A.schwarz_parameters[1] is subdomain_ptr
    A.schwarz_parameters[2] is inv_subblock
    A.schwarz_parameters[3] is inv_subblock_ptr
    '''

    # Check if A has a pre-existing set of Schwarz parameters
    if hasattr(A, 'schwarz_parameters'):
        if subdomain is not None and subdomain_ptr is not None:
            # check that the existing parameters correspond to the same
            # subdomains
            if np.array(A.schwarz_parameters[0] == subdomain).all() and \
               np.array(A.schwarz_parameters[1] == subdomain_ptr).all():
                return A.schwarz_parameters
        else:
            return A.schwarz_parameters

    # Default is to use the overlapping regions defined by A's sparsity pattern
    if subdomain is None or subdomain_ptr is None:
        subdomain_ptr = A.indptr.copy()
        subdomain = A.indices.copy()

    # Extract each subdomain's block from the matrix
    if inv_subblock is None or inv_subblock_ptr is None:
        inv_subblock_ptr = np.zeros(subdomain_ptr.shape,
                                    dtype=A.indices.dtype)
        blocksize = (subdomain_ptr[1:] - subdomain_ptr[:-1])
        inv_subblock_ptr[1:] = np.cumsum(blocksize*blocksize)

        # Extract each block column from A
        inv_subblock = np.zeros((inv_subblock_ptr[-1],), dtype=A.dtype)
        amg_core.extract_subblocks(A.indptr, A.indices, A.data, inv_subblock,
                                   inv_subblock_ptr, subdomain, subdomain_ptr,
                                   int(subdomain_ptr.shape[0]-1), A.shape[0])
        # Choose tolerance for which singular values are zero in *gelss below
        t = A.dtype.char
        eps = np.finfo(np.float).eps
        feps = np.finfo(np.single).eps
        geps = np.finfo(np.longfloat).eps
        _array_precision = {'f': 0, 'd': 1, 'g': 2, 'F': 0, 'D': 1, 'G': 2}
        cond = {0: feps*1e3, 1: eps*1e6, 2: geps*1e6}[_array_precision[t]]

        # Invert each block column
        my_pinv, = la.get_lapack_funcs(['gelss'],
                                       (np.ones((1,), dtype=A.dtype)))
        for i in range(subdomain_ptr.shape[0]-1):
            m = blocksize[i]
            rhs = sp.eye(m, m, dtype=A.dtype)
            j0 = inv_subblock_ptr[i]
            j1 = inv_subblock_ptr[i+1]
            gelssoutput = my_pinv(inv_subblock[j0:j1].reshape(m, m),
                                  rhs, cond=cond, overwrite_a=True,
                                  overwrite_b=True)
            inv_subblock[j0:j1] = np.ravel(gelssoutput[1])

    A.schwarz_parameters = (subdomain, subdomain_ptr, inv_subblock,
                            inv_subblock_ptr)
    return A.schwarz_parameters

# from pyamg.utils import dispatcher
# dispatch = dispatcher( dict([ (fn,eval(fn)) for fn in __all__ ]) )
