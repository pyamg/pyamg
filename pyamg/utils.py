__all__ = ['approximate_spectral_radius', 'infinity_norm', 'diag_sparse',
        'norm', 'profile_solver']
__all__ += ['UnAmal', 'Coord2RBM', 'BSR_Get_Row', 'BSR_Row_WriteScalar', 
        'BSR_Row_WriteVect' ]

import numpy
import scipy
from numpy import fromfile, ascontiguousarray, mat, int32, inner, dot, \
                  ravel, arange, concatenate, tile, asarray, sqrt, diff, \
                  zeros, ones, empty, asmatrix
from scipy import rand, real                  
from scipy.linalg import eigvals
from scipy.lib.blas import get_blas_funcs
from scipy.sparse import isspmatrix, isspmatrix_csr, isspmatrix_csc, \
        isspmatrix_bsr, csr_matrix, csc_matrix, bsr_matrix, coo_matrix
from scipy.sparse.sputils import upcast
from scipy.sparse.linalg import eigen, eigen_symmetric

def norm(x):
    #currently 40x faster than scipy.linalg.norm(x)
    x = ravel(x)
    return real(sqrt(inner(x,x)))

def axpy(x,y,a=1.0):
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
    """approximate the spectral radius of a matrix

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
    """
   
    if not isspmatrix(A):
        A = asmatrix(A) #convert dense arrays to matrix type
    
    if A.shape[0] != A.shape[1]:
        raise ValueError,'expected square matrix'

    maxiter = min(A.shape[0],maxiter)

    #TODO make method adaptive

    numpy.random.seed(0)  #make results deterministic

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
    A = ml.levels[0].A
    b = A * rand(A.shape[0],1)

    if accel is None:
        x_sol, residuals = ml.solve(b, return_residuals=True, **kwargs)
    else:
        residuals = []
        def callback(x):
            residuals.append( norm(ravel(b) - ravel(A*x)) )
        A.psolve = ml.psolve
        accel(A, b, callback=callback, **kwargs)

    return asarray(residuals)


def infinity_norm(A):
    """
    Infinity norm of a sparse matrix (maximum absolute row sum).  This serves
    as an upper bound on spectral radius.
    """

    if isspmatrix_csr(A) or isspmatrix_csc(A):
        #avoid copying index and ptr arrays
        abs_A = A.__class__((abs(A.data),A.indices,A.indptr),shape=A.shape)
        return (abs_A * ones(A.shape[1],dtype=A.dtype)).max()
    else:
        return (abs(A) * ones(A.shape[1],dtype=A.dtype)).max()

def diag_sparse(A):
    """
    If A is a sparse matrix (e.g. csr_matrix or csc_matrix)
       - return the diagonal of A as an array

    Otherwise
       - return a csr_matrix with A on the diagonal
    """

    #TODO integrate into SciPy?
    if isspmatrix(A):
        return A.diagonal()
    else:
        return csr_matrix((asarray(A),arange(len(A)),arange(len(A)+1)),(len(A),len(A)))

def scale_rows(A,v,copy=True):
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
#        else:
#            # otherwise, assume fn is itself a function handle
#            pass
#            #TODO check that fn is callable
#    
#        wrapped = partial(fn, **opts)
#        update_wrapper(wrapped, fn)
#    
#        return wrapped
#
#    return dispatcher



##############################################################################################
#                                           JBS Utils                                        #
##############################################################################################

def UnAmal(A, RowsPerBlock, ColsPerBlock):
    """Unamalgamate a CSR A with blocks of 1's.  
    Equivalent to Kronecker_Product(A, ones(RowsPerBlock, ColsPerBlock)

    Input
    =====
    A                   Amalmagated matrix, assumed to be in CSR format
    RowsPerBlock &
    ColsPerBlock        Give A blocks of size (RowsPerBlock, ColsPerBlock)
    
    Output
    ======
    A_UnAmal:           BSR matrix that is essentially a Kronecker product of 
                        A and ones(RowsPerBlock, ColsPerBlock

    """
    data = ones( (A.indices.shape[0], RowsPerBlock, ColsPerBlock) )
    return bsr_matrix((data, A.indices, A.indptr), shape=(RowsPerBlock*A.shape[0], ColsPerBlock*A.shape[1]) )

def Coord2RBM(numNodes, numPDEs, x, y, z):
    """Convert 2D or 3D coordinates into Rigid body modes for use as near nullspace modes in elasticity AMG solvers

    Input
    =====
    numNodes    Number of nodes
    numPDEs     Number of dofs per node
    x,y,z       Coordinate vectors


    Output
    ======
    rbm:        Matrix of size (numNodes*numPDEs) x (1 | 6) containing the 6 rigid body modes

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


############################################################################################
#                    JBS --- Define BSR helper functions                                   #
############################################################################################

def BSR_Get_Row(A, i):
    """Return row i in BSR matrix A.  Only nonzero entries are returned

    Input
    =====
    A   Matrix assumed to be in BSR format
    i   row number

    Output
    ======
    z   Actual nonzero values for row i
        colindx Array of column indices for the nonzeros of row i
    
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
    """Write a scalar at each nonzero location in row i of BSR matrix A

    Input
    =====
    A   Matrix assumed to be in BSR format
    i   row number
    x   scalar to overwrite nonzeros of row i in A

    Output
    ======
    A   All nonzeros in row i of A have been overwritten with x.  
        If x is a vector, the first length(x) nonzeros in row i 
        of A have been overwritten with entries from x

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
    """Overwrite the nonzeros in row i of BSR matrix A with the vector x.  
       length(x) and nnz(A[i,:]) must be equivalent.

    Input
    =====
    A   Matrix assumed to be in BSR format
    i   row number
    x   Array of values to overwrite nonzeros in row i of A

    Output
    ======
    A   The nonzeros in row i of A have been
        overwritten with entries from x.  x must be same
        length as nonzeros of row i.  This is guaranteed
        when this routine is used with vectors derived form
        Get_BSR_Row

    """
    
    blocksize = A.blocksize[0]
    BlockIndx = i/blocksize
    rowstart = A.indptr[BlockIndx]
    rowend = A.indptr[BlockIndx+1]
    localRowIndx = i%blocksize
    
    # This line fixes one of the idiotic things about the array/matrix setup.
    # Sometimes I really wish for the Matlab matrix "slicing" interface rather 
    # than this.
    x = x.__array__().reshape( (max(x.shape),) )

    #counter = 0
    #for j in range(rowstart, rowend):
    #   indys = A.data[j,localRowIndx,:].nonzero()[0]
    #   increment = min(indys.shape[0], blocksize)
    #   A.data[j,localRowIndx,indys] = x[counter:(counter+increment), 0]
    #   counter += increment

    indys = A.data[rowstart:rowend, localRowIndx, :].nonzero()
    A.data[rowstart:rowend, localRowIndx, :][indys[0], indys[1]] = x


###################################################################################################


