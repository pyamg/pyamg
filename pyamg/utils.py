"""General utility functions for pyamg"""

__docformat__ = "restructuredtext en"

from numpy import fromfile, ascontiguousarray, mat, int32, inner, dot, \
                  ravel, arange, concatenate, tile, asarray, sqrt, diff, \
                  zeros, ones, empty, asmatrix, array, random, rank
from scipy import rand, real                  
from scipy.linalg import eigvals, norm
from scipy.lib.blas import get_blas_funcs
from scipy.sparse import isspmatrix, isspmatrix_csr, isspmatrix_csc, \
        isspmatrix_bsr, csr_matrix, csc_matrix, bsr_matrix, coo_matrix
from scipy.sparse.sputils import upcast
from scipy.sparse.linalg import eigen, eigen_symmetric

__all__ = ['approximate_spectral_radius', 'infinity_norm', 'diag_sparse',
        'norm', 'profile_solver']
__all__ += ['UnAmal', 'Coord2RBM', 'BSR_Get_Row', 'BSR_Row_WriteScalar', 
        'BSR_Row_WriteVect' ]


def norm(x):
    """
    2-norm of a vector
    
    Parameters
    ----------
    x : array_like
        Vector of complex or real values

    Return
    ------
    n : float
        2-norm of a vector

    Notes
    -----
    - currently 40x faster than scipy.linalg.norm(x), which calls
      sqrt(numpy.sum(real((conjugate(x)*x)),axis=0)) resulting in an extra copy
    - only handles the 2-norm for vectors

    See Also
    --------
    scipy.linalg.norm : scipy general matrix or vector norm
    """

    x = ravel(x)
    return real(sqrt(inner(x,x)))

def axpy(x,y,a=1.0):
    """
    Quick level-1 call to blas::
    y = a*x+y

    Parameters
    ----------
    x : array_like
        nx1 real or complex vector
    y : array_like
        nx1 real or complex vector
    a : float
        real or complex scalar

    Return
    ------
    y : array_like
        Input variable y is rewritten

    Notes
    -----
    The call to get_blas_funcs automatically determines the prefix for the blas
    call.
    """
    fn = get_blas_funcs(['axpy'], [x,y])[0]
    fn(x,y,a)


#def approximate_spectral_radius(A, tol=0.1, maxiter=10, symmetric=False):
#    """approximate the spectral radius of a matrix
#
#    Parameters
#    ----------
#
#    A : {dense or sparse matrix}
#        E.g. csr_matrix, csc_matrix, ndarray, etc.
#    tol : {scalar}
#        Tolerance of approximation
#    maxiter : {integer}
#        Maximum number of iterations to perform
#    symmetric : {boolean}
#        True if A is symmetric, False otherwise (default)
#
#    Returns
#    -------
#        An approximation to the spectral radius of A
#
#    """
#    if symmetric:
#        method = eigen_symmetric
#    else:
#        method = eigen
#    
#    return norm( method(A, k=1, tol=0.1, which='LM', maxiter=maxiter, return_eigenvectors=False) )


def approximate_spectral_radius(A,tol=0.1,maxiter=10,symmetric=None):
    """
    Approximate the spectral radius of a matrix

    Parameters
    ----------

    A : {dense or sparse matrix}
        E.g. csr_matrix, csc_matrix, ndarray, etc.
    tol : {scalar}
        Tolerance of approximation
    maxiter : {integer}
        Maximum number of iterations to perform
    symmetric : {boolean}
        True  - if A is symmetric
                Lanczos iteration is used (more efficient)
        False - if A is non-symmetric (default
                Arnoldi iteration is used (less efficient)

    Returns
    -------
    An approximation to the spectral radius of A

    Notes
    -----
    The spectral radius is approximated by looking at the Ritz eigenvalues.
    Arnoldi iteration (or Lanczos) is used to project the matrix A onto a
    Krylov subspace: H = Q* A Q.  The eigenvalues of H (i.e. the Ritz
    eigenvalues) should represent the eigenvalues of A in the sense that the
    minimum and maximum values are usually well matched (for the symmetric case
    it is true since the eigenvalues are real).

    References
    ----------
    Z. Bai, J. Demmel, J. Dongarra, A. Ruhe, and H. van der Vorst, editors.
    "Templates for the Solution of Algebraic Eigenvalue Problems: A Practical
    Guide", SIAM, Philadelphia, 2000.

    Examples
    --------
    >>> from pyamg.utils import approximate_spectral_radius
    >>> from scipy import rand
    >>> from scipy.linalg import eigvals, norm
    >>> A = rand(10,10)
    >>> print approximate_spectral_radius(A,maxiter=3)
    >>> print max([norm(x) for x in eigvals(A)])

    TODO
    ----
    Make the method adaptive (restarts)
    """
   
    if type(A) == type( array([0.0]) ):
        A = asmatrix(A) #convert dense arrays to matrix type
    
    if A.shape[0] != A.shape[1]:
        raise ValueError,'expected square matrix'

    maxiter = min(A.shape[0],maxiter)

    random.seed(0)  #make results deterministic

    v0  = rand(A.shape[1],1)
    v0 /= norm(v0)

    H  = zeros((maxiter+1,maxiter))
    V = [v0]

    for j in range(maxiter):
        w = A * V[-1]
   
        if symmetric:
            if j >= 1:
                H[j-1,j] = beta
                w -= beta * V[-2]

            alpha = dot(ravel(w),ravel(V[-1]))
            H[j,j] = alpha
            w -= alpha * V[-1]  #axpy(V[-1],w,-alpha) 
            
            beta = norm(w)
            H[j+1,j] = beta

            if (H[j+1,j] < 1e-10): break
            
            w /= beta

            V.append(w)
            V = V[-2:] #retain only last two vectors

        else:
            #orthogonalize against Vs
            for i,v in enumerate(V):
                H[i,j] = dot(ravel(w),ravel(v))
                w -= H[i,j]*v #axpy(v,w,-H[i,j])
            H[j+1,j] = norm(w)
            if (H[j+1,j] < 1e-10): break
            
            w /= H[j+1,j] 
            V.append(w)
   
            # if upper 2x2 block of Hessenberg matrix H is almost symmetric,
            # and the user has not explicitly specified symmetric=False,
            # then switch to symmetric Lanczos algorithm
            #if symmetric is not False and j == 1:
            #    if abs(H[1,0] - H[0,1]) < 1e-12:
            #        #print "using symmetric mode"
            #        symmetric = True
            #        V = V[1:]
            #        H[1,0] = H[0,1]
            #        beta = H[2,1]
    
    #print "Approximated spectral radius in %d iterations" % (j + 1)

    return max([norm(x) for x in eigvals(H[:j+1,:j+1])])      


def profile_solver(ml, accel=None, **kwargs):
    """
    A quick solver to profile a particular multilevel object

    Parameters
    ----------
    ml : multilevel
        Fully constructed multilevel object
    accel : function pointer
        Pointer to a valid Krylov solver (e.g. gmres, cg)

    Returns
    -------
    residuals : array
        Array of residuals for each iteration

    See Also
    --------
    multilevel.psolve, multilevel.solve

    Examples
    --------
    >>> from numpy import ones
    >>> from scipy.sparse import spdiags
    >>> from scipy.sparse.linalg import cg
    >>> from pyamg.classical import ruge_stuben_solver
    >>> from pyamg.utils import profile_solver
    >>> n=100
    >>> e = ones((n,1)).ravel()
    >>> data = [ -1*e, 2*e, -1*e ]
    >>> A = spdiags(data,[-1,0,1],n,n)
    >>> b = A*ones(A.shape[0])
    >>> ml = ruge_stuben_solver(A, max_coarse=10)
    >>> res = profile_solver(ml,accel=cg)
    >>> print res
    """
    A = ml.levels[0].A
    b = A * rand(A.shape[0],1)
    residuals = []

    if accel is None:
        x_sol = ml.solve(b, residuals=residuals, **kwargs)
    else:
        def callback(x):
            residuals.append( norm(ravel(b) - ravel(A*x)) )
        M = ml.aspreconditioner(cycle=kwargs.get('cycle','V'))
        accel(A, b, M=M, callback=callback, **kwargs)

    return asarray(residuals)


def infinity_norm(A):
    """
    Infinity norm of a matrix (maximum absolute row sum).  

    Parameters
    ----------
    A : csr_matrix, csc_matrix, sparse, or numpy matrix
        Sparse or dense matrix
    
    Returns
    -------
    n : float
        Infinity norm of the matrix
    
    Notes
    -----
    - This serves as an upper bound on spectral radius.
    - csr and csc avoid a deep copy
    - dense calls scipy.linalg.norm

    See Also
    --------
    scipy.linalg.norm : dense matrix norms

    Examples
    --------
    >>> from numpy import ones
    >>> from scipy.sparse import spdiags
    >>> from pyamg.utils import infinity_norm
    >>> n=100
    >>> e = ones((n,1)).ravel()
    >>> data = [ -1*e, 2*e, -1*e ]
    >>> A = spdiags(data,[-1,0,1],n,n)
    >>> print infinity_norm(A)
    """

    if isspmatrix_csr(A) or isspmatrix_csc(A):
        #avoid copying index and ptr arrays
        abs_A = A.__class__((abs(A.data),A.indices,A.indptr),shape=A.shape)
        return (abs_A * ones((A.shape[1]),dtype=A.dtype)).max()
    elif isspmatrix(A):
        return (abs(A) * ones((A.shape[1]),dtype=A.dtype)).max()
    else:
        return norm(A,inf)

def diag_sparse(A):
    """
    If A is a sparse matrix (e.g. csr_matrix or csc_matrix)
       - return the diagonal of A as an array

    Otherwise
       - return a csr_matrix with A on the diagonal

    Parameters
    ----------
    A : sparse matrix or rank 1 array
        General sparse matrix or array of diagonal entries

    Returns
    -------
    B : array or sparse matrix
        Diagonal sparse is returned as csr if A is dense otherwise return an
        array of the diagonal

    Examples
    --------
    >>> from scipy import rand
    >>> from pyamg.utils import diag_sparse
    >>> d = rand(6,1).ravel()
    >>> print diag_sparse(d).todense()
    """
    if isspmatrix(A):
        return A.diagonal()
    else:
        if(rank(A)!=1):
            raise ValueError,'input diagonal array expected to be rank 1'
        return csr_matrix((asarray(A),arange(len(A)),arange(len(A)+1)),(len(A),len(A)))

def scale_rows(A,v,copy=True):
    """
    Scale the sparse rows of a matrix

    Parameters
    ----------
    A : sparse matrix
        Sparse matrix with M rows
    v : array_like
        Array of M scales
    copy : {True,False}
        - If copy=True, then the matrix is copied to a new and different return
          matrix (e.g. B=scale_rows(A,v))
        - If copy=False, then the matrix is overwritten deeply (e.g.
          scale_rows(A,v,copy=False) overwrites A)

    Returns
    -------
    A : sparse matrix
        Scaled sparse matrix in original format

    See Also
    --------
    scipy.sparse.sparsetools.csr_scale_rows, scale_columns

    Notes
    -----
    - if A is a csc_matrix, the transpose A.T is passed to scale_columns
    - if A is not csr, csc, or bsr, it is converted to csr and sent to scale_rows

    Examples
    --------
    >>> from numpy import ones
    >>> from scipy.sparse import spdiags
    >>> from pyamg.utils import scale_rows
    >>> n=5
    >>> e = ones((n,1)).ravel()
    >>> data = [ -1*e, 2*e, -1*e ]
    >>> A = spdiags(data,[-1,0,1],n,n-1).tocsr()
    >>> print scale_rows(A,5*ones((A.shape[0],1))).todense()
    """
    from scipy.sparse.sparsetools import csr_scale_rows, bsr_scale_rows

    v = ravel(v)

    if isspmatrix_csr(A) or isspmatrix_bsr(A):
        M,N = A.shape
        if M != len(v):
            raise ValueError,'scale vector has incompatible shape'

        if copy:
            A = A.copy()
            A.data = asarray(A.data,dtype=upcast(A.dtype,v.dtype))
        else:
            v = asarray(v,dtype=A.dtype)

        if isspmatrix_csr(A):
            csr_scale_rows(M, N, A.indptr, A.indices, A.data, v)
        else:
            R,C = A.blocksize
            bsr_scale_rows(M/R, N/C, R, C, A.indptr, A.indices, ravel(A.data), v)

        return A
    elif isspmatrix_csc(A):
        return scale_columns(A.T,v)
    else:
        return scale_rows(csr_matrix(A),v)
        
def scale_columns(A,v,copy=True):
    """
    Scale the sparse columns of a matrix

    Parameters
    ----------
    A : sparse matrix
        Sparse matrix with N rows
    v : array_like
        Array of N scales
    copy : {True,False}
        - If copy=True, then the matrix is copied to a new and different return
          matrix (e.g. B=scale_columns(A,v))
        - If copy=False, then the matrix is overwritten deeply (e.g.
          scale_columns(A,v,copy=False) overwrites A)

    Returns
    -------
    A : sparse matrix
        Scaled sparse matrix in original format

    See Also
    --------
    scipy.sparse.sparsetools.csr_scale_columns, scale_rows

    Notes
    -----
    - if A is a csc_matrix, the transpose A.T is passed to scale_rows
    - if A is not csr, csc, or bsr, it is converted to csr and sent to scale_rows

    Examples
    --------
    >>> from numpy import ones
    >>> from scipy.sparse import spdiags
    >>> from pyamg.utils import scale_columns
    >>> n=5
    >>> e = ones((n,1)).ravel()
    >>> data = [ -1*e, 2*e, -1*e ]
    >>> A = spdiags(data,[-1,0,1],n,n-1).tocsr()
    >>> print scale_columns(A,5*ones((A.shape[1],1))).todense()
    """
    from scipy.sparse.sparsetools import csr_scale_columns, bsr_scale_columns

    v = ravel(v)

    if isspmatrix_csr(A) or isspmatrix_bsr(A):
        M,N = A.shape
        if N != len(v):
            raise ValueError,'scale vector has incompatible shape'

        if copy:
            A = A.copy()
            A.data = asarray(A.data,dtype=upcast(A.dtype,v.dtype))
        else:
            v = asarray(v,dtype=A.dtype)

        if isspmatrix_csr(A):
            csr_scale_columns(M, N, A.indptr, A.indices, A.data, v)
        else:
            R,C = A.blocksize
            bsr_scale_columns(M/R, N/C, R, C, A.indptr, A.indices, ravel(A.data), v)

        return A
    elif isspmatrix_csc(A):
        return scale_rows(A.T,v)
    else:
        return scale_rows(csr_matrix(A),v)

def symmetric_rescaling(A,copy=True):
    """
    Scale the matrix symmetrically::

        A = D^{-1/2} A D^{-1/2}

    where D=diag(A).

    The left multiplication is accomplished through scale_rows and the right
    multiplication is done through scale columns.

    Parameters
    ----------
    A : sparse matrix
        Sparse matrix with N rows
    copy : {True,False}
        - If copy=True, then the matrix is copied to a new and different return
          matrix (e.g. B=symmetric_rescaling(A))
        - If copy=False, then the matrix is overwritten deeply (e.g.
          symmetric_rescaling(A,copy=False) overwrites A)

    Returns
    -------
    D_sqrt : array
        Array of sqrt(diag(A))
    D_sqrt_inv : array
        Array of 1/sqrt(diag(A))
    DAD    : csr_matrix
        Symmetrically scaled A

    Notes
    -----
    - if A is not csr, it is converted to csr and sent to scale_rows

    Examples
    --------
    >>> from numpy import ones
    >>> from scipy.sparse import spdiags
    >>> from pyamg.utils import symmetric_rescaling
    >>> n=5
    >>> e = ones((n,1)).ravel()
    >>> data = [ -1*e, 2*e, -1*e ]
    >>> A = spdiags(data,[-1,0,1],n,n).tocsr()
    >>> Ds, Dsi, DAD = symmetric_rescaling(A)
    >>> print DAD.todense()
    """
    if isspmatrix_csr(A) or isspmatrix_csc(A) or isspmatrix_bsr(A):
        if A.shape[0] != A.shape[1]:
            raise ValueError,'expected square matrix'

        D = diag_sparse(A)
        mask = D == 0

        D_sqrt = sqrt(abs(D))
        D_sqrt_inv = 1.0/D_sqrt
        D_sqrt_inv[mask] = 0

        DAD = scale_rows(A,D_sqrt_inv,copy=copy)
        DAD = scale_columns(DAD,D_sqrt_inv,copy=False)

        return D_sqrt,D_sqrt_inv,DAD

    else:
        return symmetric_rescaling(csr_matrix(A))


#from functools import partial, update_wrapper
#def dispatcher(name_to_handle):
#    def dispatcher(arg):
#        if isinstance(arg,tuple):
#            fn,opts = arg[0],arg[1]
#        else:
#            fn,opts = arg,{}
#    
#        if fn in name_to_handle:
#            # convert string into function handle
#            fn = name_to_handle[fn] 
#        #elif isinstance(fn, type(ones)):
#        #    pass     
#        elif callable(fn):
#            # if fn is itself a function handle
#            pass
#        else:
#            raise TypeError('Expected function')
#
#        wrapped = partial(fn, **opts)
#        update_wrapper(wrapped, fn)
#    
#        return wrapped
#
#    return dispatcher


def UnAmal(A, RowsPerBlock, ColsPerBlock):
    """
    Unamalgamate a CSR A with blocks of 1's.  

    Equivalent to Kronecker_Product(A, ones(RowsPerBlock, ColsPerBlock)

    Parameters
    ----------
    A : csr_matrix
        Amalgamted matrix
    RowsPerBlock : int
        Give A blocks of size (RowsPerBlock, ColsPerBlock)
    ColsPerBlock : int
        Give A blocks of size (RowsPerBlock, ColsPerBlock)
    
    Returns
    -------
    A_UnAmal : bsr_matrix 
        Similar to a Kronecker product of A and ones(RowsPerBlock, ColsPerBlock)

    Examples
    --------
    >>> from numpy import array
    >>> from scipy.sparse import csr_matrix
    >>> from pyamg.utils import UnAmal
    >>> row = array([0,0,1,2,2,2])
    >>> col = array([0,2,2,0,1,2])
    >>> data = array([1,2,3,4,5,6])
    >>> A = csr_matrix( (data,(row,col)), shape=(3,3) )
    >>> A.todense()
    >>> UnAmal(A,2,2).todense()
    """
    data = ones( (A.indices.shape[0], RowsPerBlock, ColsPerBlock) )
    return bsr_matrix((data, A.indices, A.indptr), shape=(RowsPerBlock*A.shape[0], ColsPerBlock*A.shape[1]) )

def Coord2RBM(numNodes, numPDEs, x, y, z):
    """
    Convert 2D or 3D coordinates into Rigid body modes for use as near
    nullspace modes in elasticity AMG solvers

    Parameters
    ----------
    numNodes : int
        Number of nodes
    numPDEs : 
        Number of dofs per node
    x,y,z : array_like
        Coordinate vectors

    Returns
    -------
    rbm : matrix 
        A matrix of size (numNodes*numPDEs) x (1 | 6) containing the 6 rigid
        body modes

    Examples
    --------
    >>> from pyamg.utils import Coord2RBM
    >>> Coord2RBM(3,6,array([0,1,2]),array([0,1,2]),array([0,1,2]))
    """

    #check inputs
    if(numPDEs == 1):
        numcols = 1
    elif( (numPDEs == 3) or (numPDEs == 6) ):
        numcols = 6
    else:
        raise ValueError("Coord2RBM(...) only supports 1, 3 or 6 PDEs per spatial location, i.e. numPDEs = [1 | 3 | 6].  You've entered " \
                + str(numPDEs) + "." )

    if( (max(x.shape) != numNodes) or (max(y.shape) != numNodes) or (max(z.shape) != numNodes) ):
        raise ValueError("Coord2RBM(...) requires coordinate vectors of equal length.  Length must be numNodes = " + str(numNodes)) 

    #if( (min(x.shape) != 1) or (min(y.shape) != 1) or (min(z.shape) != 1) ):
    #    raise ValueError("Coord2RBM(...) requires coordinate vectors that are (numNodes x 1) or (1 x numNodes).") 


    #preallocate rbm
    rbm = mat(zeros((numNodes*numPDEs, numcols)))
    
    for node in range(numNodes):
        dof = node*numPDEs

        if(numPDEs == 1):
            rbm[node] = 1.0 
                
        if(numPDEs == 6): 
            for ii in range(3,6):        #lower half = [ 0 I ]
                for jj in range(0,6):
                    if(ii == jj):
                        rbm[dof+ii, jj] = 1.0 
                    else: 
                        rbm[dof+ii, jj] = 0.0

        if((numPDEs == 3) or (numPDEs == 6) ): 
            for ii in range(0,3):        #upper left = [ I ]
                for jj in range(0,3):
                    if(ii == jj):
                        rbm[dof+ii, jj] = 1.0 
                    else: 
                        rbm[dof+ii, jj] = 0.0

            for ii in range(0,3):        #upper right = [ Q ]
                for jj in range(3,6):
                    if( ii == (jj-3) ):
                        rbm[dof+ii, jj] = 0.0
                    else:
                        if( (ii+jj) == 4):
                            rbm[dof+ii, jj] = z[node]
                        elif( (ii+jj) == 5 ): 
                            rbm[dof+ii, jj] = y[node]
                        elif( (ii+jj) == 6 ): 
                            rbm[dof+ii, jj] = x[node]
                        else:
                            rbm[dof+ii, jj] = 0.0
            
            ii = 0 
            jj = 5 
            rbm[dof+ii, jj] *= -1.0
    
            ii = 1 
            jj = 3 
            rbm[dof+ii, jj] *= -1.0
    
            ii = 2 
            jj = 4 
            rbm[dof+ii, jj] *= -1.0
    
    return rbm

def BSR_Get_Row(A, i):
    """
    Return row i in BSR matrix A.  Only nonzero entries are returned

    Parameters
    ----------
    A : bsr_matrix
        Input matrix
    i : int
        Row number

    Returns
    -------
    z : array
        Actual nonzero values for row i colindx Array of column indices for the
        nonzeros of row i
   
    Examples
    --------
    >>> from numpy import array
    >>> from scipy.sparse import bsr_matrix
    >>> from pyamg.utils import BSR_Get_Row
    >>> indptr  = array([0,2,3,6])
    >>> indices = array([0,2,2,0,1,2])
    >>> data    = array([1,2,3,4,5,6]).repeat(4).reshape(6,2,2)
    >>> B = bsr_matrix( (data,indices,indptr), shape=(6,6) )
    >>> BSR_Get_Row(B,2)
    """
    
    blocksize = A.blocksize[0]
    BlockIndx = i/blocksize
    rowstart = A.indptr[BlockIndx]
    rowend = A.indptr[BlockIndx+1]
    localRowIndx = i%blocksize

    #Get z
    indys = A.data[rowstart:rowend, localRowIndx, :].nonzero()
    z = A.data[rowstart:rowend, localRowIndx, :][indys[0], indys[1]]


    colindx = zeros((1, z.__len__()), dtype=int32)
    counter = 0

    for j in range(rowstart, rowend):
        coloffset = blocksize*A.indices[j]
        indys = A.data[j,localRowIndx,:].nonzero()[0]
        increment = indys.shape[0]
        colindx[0,counter:(counter+increment)] = coloffset + indys
        counter += increment

    return mat(z).T, colindx[0,:]

def BSR_Row_WriteScalar(A, i, x): 
    """
    Write a scalar at each nonzero location in row i of BSR matrix A

    Parameters
    ----------
    A : bsr_matrix
        Input matrix
    i : int
        Row number
    x : float
        Scalar to overwrite nonzeros of row i in A

    Returns
    -------
    A : bsr_matrix
        All nonzeros in row i of A have been overwritten with x.  
        If x is a vector, the first length(x) nonzeros in row i 
        of A have been overwritten with entries from x

    Examples
    --------
    >>> from numpy import array
    >>> from scipy.sparse import bsr_matrix
    >>> from pyamg.utils import BSR_Row_WriteScalar
    >>> indptr  = array([0,2,3,6])
    >>> indices = array([0,2,2,0,1,2])
    >>> data    = array([1,2,3,4,5,6]).repeat(4).reshape(6,2,2)
    >>> B = bsr_matrix( (data,indices,indptr), shape=(6,6) )
    >>> BSR_Row_WriteScalar(B,5,22)
    """
    
    blocksize = A.blocksize[0]
    BlockIndx = i/blocksize
    rowstart = A.indptr[BlockIndx]
    rowend = A.indptr[BlockIndx+1]
    localRowIndx = i%blocksize

    #for j in range(rowstart, rowend):
    #   indys = A.data[j,localRowIndx,:].nonzero()[0]
    #   increment = indys.shape[0]
    #   A.data[j,localRowIndx,indys] = x
    
    indys = A.data[rowstart:rowend, localRowIndx, :].nonzero()
    A.data[rowstart:rowend, localRowIndx, :][indys[0], indys[1]] = x


def BSR_Row_WriteVect(A, i, x): 
    """
    Overwrite the nonzeros in row i of BSR matrix A with the vector x.  
    length(x) and nnz(A[i,:]) must be equivalent.

    Parameters
    ----------
    A : bsr_matrix
        Matrix assumed to be in BSR format
    i : int
        Row number
    x : array
        Array of values to overwrite nonzeros in row i of A

    Returns
    -------
    A : bsr_matrix
        The nonzeros in row i of A have been
        overwritten with entries from x.  x must be same
        length as nonzeros of row i.  This is guaranteed
        when this routine is used with vectors derived form
        Get_BSR_Row

    Examples
    --------
    >>> from numpy import array
    >>> from scipy.sparse import bsr_matrix
    >>> from pyamg.utils import BSR_Row_WriteVect
    >>> indptr  = array([0,2,3,6])
    >>> indices = array([0,2,2,0,1,2])
    >>> data    = array([1,2,3,4,5,6]).repeat(4).reshape(6,2,2)
    >>> B = bsr_matrix( (data,indices,indptr), shape=(6,6) )
    >>> BSR_Row_WriteVect(B,5,array([11,22,33,44,55,66]))
    """
    
    blocksize = A.blocksize[0]
    BlockIndx = i/blocksize
    rowstart = A.indptr[BlockIndx]
    rowend = A.indptr[BlockIndx+1]
    localRowIndx = i%blocksize
    
    # like matlab slicing:
    x = x.__array__().reshape( (max(x.shape),) )

    #counter = 0
    #for j in range(rowstart, rowend):
    #   indys = A.data[j,localRowIndx,:].nonzero()[0]
    #   increment = min(indys.shape[0], blocksize)
    #   A.data[j,localRowIndx,indys] = x[counter:(counter+increment), 0]
    #   counter += increment

    indys = A.data[rowstart:rowend, localRowIndx, :].nonzero()
    A.data[rowstart:rowend, localRowIndx, :][indys[0], indys[1]] = x
