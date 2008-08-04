"""Relaxation methods for linear systems"""

__docformat__ = "restructuredtext en"

from warnings import warn

from numpy import empty_like, asarray, arange, ravel, ones_like, zeros

import multigridtools
from scipy.sparse import isspmatrix_csr, isspmatrix_csc, isspmatrix_bsr, \
        csr_matrix, coo_matrix, bsr_matrix, SparseEfficiencyWarning

__all__ = ['sor', 'gauss_seidel', 'jacobi', 'polynomial', 'kaczmarz_jacobi', 'kaczmarz_richardson', 'kaczmarz_gauss_seidel']
__all__ += ['gauss_seidel_indexed'] #TODO remove

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
    When omega=1.0, then SOR is equivalent to Gauss-Seidel.

    """
    x_old = empty_like(x)

    for i in range(iterations):
        x_old[:] = x
        gauss_seidel(A,x,b,iterations=1,sweep=sweep)

        x     *= omega
        x_old *= (1-omega)
        x     += x_old


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

    """
    
    #TODO add support for block GS on BSR format
    x = ravel(x) #TODO warn if not inplace
    b = ravel(b)

    if isspmatrix_csr(A):
        pass
    elif isspmatrix_bsr(A):
        R,C = A.blocksize
        if R != C:
            raise ValueError('BSR blocks must be square')
    else:
        warn('implicit conversion to CSR', SparseEfficiencyWarning)
        A = csr_matrix(A)

    if A.shape[0] != A.shape[1]:
        raise ValueError('expected square matrix')

    if A.shape[1] != len(x) or len(x) != len(b):
        raise ValueError('unexpected number of unknowns')

    
    if sweep == 'forward':
        row_start,row_stop,row_step = 0,len(x),1
    elif sweep == 'backward':
        row_start,row_stop,row_step = len(x)-1,-1,-1 
    elif sweep == 'symmetric':
        for iter in xrange(iterations):
            gauss_seidel(A,x,b,iterations=1,sweep='forward')
            gauss_seidel(A,x,b,iterations=1,sweep='backward')
        return
    else:
        raise ValueError("valid sweep directions are 'forward', 'backward', and 'symmetric'")


    if isspmatrix_csr(A):
        for iter in xrange(iterations):
            multigridtools.gauss_seidel(A.indptr, A.indices, A.data,
                                        x, b,
                                        row_start, row_stop, row_step)
    else:
        blocksize = A.blocksize[0]
        row_start = row_start/blocksize
        row_stop  = row_stop/blocksize
        for iter in xrange(iterations):
            multigridtools.block_gauss_seidel(A.indptr, A.indices, ravel(A.data),
                                              x, b,
                                              row_start, row_stop, row_step,
                                              blocksize)


def jacobi(A, x, b, iterations=1, omega=1.0):
    """Perform Jacobi iteration on the linear system Ax=b

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
   
    """
    x = asarray(x).reshape(-1)
    b = asarray(b).reshape(-1)

    if isspmatrix_csr(A):
        pass
    else:
        warn('implicit conversion to CSR', SparseEfficiencyWarning)
        A = csr_matrix(A)


    sweep = slice(None)
    (row_start,row_stop,row_step) = sweep.indices(A.shape[0])

    if (row_stop - row_start) * row_step <= 0:  #no work to do
        return

    temp = empty_like(x)

    for iter in xrange(iterations):
        multigridtools.jacobi(A.indptr, A.indices, A.data,
                              x, b, temp,
                              row_start, row_stop, row_step,
                              omega)


def polynomial(A, x, b, coeffs):
    """Apply a polynomial smoother to the system Ax=b


    Parameters
    ----------
    A : {csr_matrix, bsr_matrix}
        Sparse NxN matrix
    x : ndarray
        Approximate solution (length N)
    b : ndarray
        Right-hand side (length N)
    coeffs : {array_like}
        Coefficients of the polynomial.  See Notes section for details.
    iterations : int
        Number of iterations to perform
    sweep : {'forward','backward','symmetric'}
        Direction of sweep


    Returns
    -------
    Nothing, x will be modified in place.


    Notes
    -----
    The smoother has the form  x[:] = x + p(A) (b - A*x) where p(A) is a 
    polynomial in A whose scalar coeffients are specified (in decending 
    order) by argument coeffs.

    - Richardson iteration p(A) = c_0:
        polynomial_smoother(A, x, b, [c_0])

    - Linear smoother p(A) = c_1*A + c_0:
        polynomial_smoother(A, x, b, [c_1,c_0])

    - Quadratic smoother p(A) = c_2*A^2 + c_1*A + c_0:
        polynomial_smoother(A, x, b, [c_2,c_1,c_0])

    For efficiency, Horner's Rule is applied to avoid computing A^k directly.

    """

    #TODO skip first matvec if x is all zero

    residual = (b - A*x)
    h = coeffs[0]*residual

    for c in coeffs[1:]:
        h = c*residual + A*h

    x += h


#TODO unify indexed with normal GS
def gauss_seidel_indexed(A, x, b, iterations=1, Id=None, sweep='forward'):
    """
    Perform Gauss-Seidel iteration on the linear system Ax=b

     Input:
         A - NxN csr_matrix
         x - rank 1 ndarray of length N
         b - rank 1 ndarray of length N
     Optional:
         iterations - number of iterations to perform (default: 1)
         Id - index list to sweep over (default: None = all nodes)
         sweep      - direction of sweep:
                        'forward' (default), 'backward', or 'symmetric'
    """

    x = ravel(x) #TODO warn if not inplace
    b = ravel(b)

    if isspmatrix_csr(A):
        pass
    else:
        warn('implicit conversion to CSR',SparseEfficiencyWarning)
        A = csr_matrix(A)

    if A.shape[0] != A.shape[1]:
        raise ValueError,'expected square matrix'

    if A.shape[1] != len(x) or len(x) != len(b):
        raise ValueError,'unexpected number of unknowns'

    # Id==none is the same as standard gauss-seidel
    if Id==None:
        gauss_seidel(A, x, b, iterations, sweep)
        return

    if sweep == 'forward':
        row_start,row_stop,row_step = 0,len(Id),1
    elif sweep == 'backward':
        row_start,row_stop,row_step = len(Id)-1,-1,-1 
    elif sweep == 'symmetric':
        for iter in xrange(iterations):
            gauss_seidel_indexed(A,x,b,iterations=1,Id=Id,sweep='forward')
            gauss_seidel_indexed(A,x,b,iterations=1,Id=Id,sweep='backward')
        return
    else:
        raise ValueError,'valid sweep directions are \'forward\', \'backward\', and \'symmetric\''

    for iter in xrange(iterations):
        multigridtools.gauss_seidel_indexed(
                            A.indptr, A.indices, A.data,
                            x, b, Id,
                            row_start, row_stop, row_step)

def kaczmarz_jacobi(A, x, b, iterations=1, omega=1.0):
    """Perform Kaczmarz Jacobi iterations on the linear system A A^T x = A^Tb

    
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

    Returns
    -------
    Nothing, x will be modified in place.
   
    """
    x = asarray(x).reshape(-1)
    b = asarray(b).reshape(-1)
    
    sweep = slice(None)
    (row_start,row_stop,row_step) = sweep.indices(A.shape[0])
    
    if isspmatrix_csr(A):
        pass
    elif isspmatrix_bsr(A):
        if A.blocksize == (1,1):
            A = csr_matrix((ravel(A.data), A.indices, A.indptr), shape=A.shape)
        else: 
            warn('implicit conversion to CSR', SparseEfficiencyWarning)
            A = csr_matrix(A)
    else:
        warn('implicit conversion to CSR', SparseEfficiencyWarning)
        A = csr_matrix(A)
    
    temp = zeros(x.shape)
    AsqRowsum = (A.multiply(A))*ones_like(x)
    for i in range(iterations):
        delta = ravel(asarray((b - A*x))/asarray(AsqRowsum))
        multigridtools.kaczmarz_jacobi(A.indptr, A.indices, A.data,
                                       x, b, delta, temp, row_start,
                                       row_stop, row_step, omega)  

def kaczmarz_richardson(A, x, b, iterations=1, omega=1.0):
    """Perform Kaczmarz Richardson iterations on the linear system A A^T x = A^Tb

    
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

    Returns
    -------
    Nothing, x will be modified in place.
   
    """
    #from pyamg.utils import approximate_spectral_radius
    #rho = approximate_spectral_radius(A)
    #omega = omega/(rho*rho)

    x = asarray(x).reshape(-1)
    b = asarray(b).reshape(-1)
    
    sweep = slice(None)
    (row_start,row_stop,row_step) = sweep.indices(A.shape[0])
    
    if isspmatrix_csr(A):
        pass
    elif isspmatrix_bsr(A):
        if A.blocksize == (1,1):
            A = csr_matrix((ravel(A.data), A.indices, A.indptr), shape=A.shape)
        else: 
            warn('implicit conversion to CSR', SparseEfficiencyWarning)
            A = csr_matrix(A)
    else:
        warn('implicit conversion to CSR', SparseEfficiencyWarning)
        A = csr_matrix(A)
    
    temp = zeros(x.shape)
    for i in range(iterations):
        delta = ravel(asarray((b - A*x)))
        multigridtools.kaczmarz_jacobi(A.indptr, A.indices, A.data,
                                           x, b, delta, temp, row_start,
                                           row_stop, row_step, omega)

def kaczmarz_gauss_seidel(A, x, b, iterations=1, sweep='forward'):
    """Perform Kaczmarz GaussSeidel iterations on the linear system A A^T x = A^Tb

    
    Parameters
    ----------
    A : {csr_matrix, bsr_matrix}
        Sparse NxN matrix
    x : { ndarray }
        Approximate solution (length N)
    b : { ndarray }
        Right-hand side (length N)
    iterations : { int }
        Number of iterations to perform
    sweep : { string }

    Returns
    -------
    Nothing, x will be modified in place.
   
    """
    
    x = asarray(x).reshape(-1)
    b = asarray(b).reshape(-1)
    

    if sweep == 'forward':
        row_start,row_stop,row_step = 0,len(x),1
    elif sweep == 'backward':
        row_start,row_stop,row_step = len(x)-1,-1,-1 
    elif sweep == 'symmetric':
        for iter in xrange(iterations):
            kaczmarz_gauss_seidel(A,x,b,iterations=1,sweep='forward')
            kaczmarz_gauss_seidel(A,x,b,iterations=1,sweep='backward')
        return
    else:
        raise ValueError("valid sweep directions are 'forward', 'backward', and 'symmetric'")

    if isspmatrix_csr(A):
        pass
    elif isspmatrix_bsr(A):
        if A.blocksize == (1,1):
            A = csr_matrix((ravel(A.data), A.indices, A.indptr), shape=A.shape)
        else: 
            warn('implicit conversion to CSR', SparseEfficiencyWarning)
            A = csr_matrix(A)
    else:
        warn('implicit conversion to CSR', SparseEfficiencyWarning)
        A = csr_matrix(A)
        
    AsqRowsum = (A.multiply(A))*ones_like(x)
    for i in range(iterations):
        multigridtools.kaczmarz_gauss_seidel(A.indptr, A.indices, A.data,
                                           x, b, row_start,
                                           row_stop, row_step, AsqRowsum)


#from pyamg.utils import dispatcher
#dispatch = dispatcher( dict([ (fn,eval(fn)) for fn in __all__ ]) )
