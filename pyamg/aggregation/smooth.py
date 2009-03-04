"""Methods to smooth tentative prolongation operators"""

__docformat__ = "restructuredtext en"

import numpy
from scipy.sparse import csr_matrix, isspmatrix_csr, bsr_matrix, isspmatrix_bsr
from scipy.linalg import pinv2

from pyamg.util.utils import scale_rows, get_diagonal
from pyamg.util.linalg import approximate_spectral_radius
from pyamg.util.utils import UnAmal
import pyamg.amg_core

__all__ = ['jacobi_prolongation_smoother', 'richardson_prolongation_smoother', 
        'energy_prolongation_smoother', 'kaczmarz_richardson_prolongation_smoother',
        'kaczmarz_jacobi_prolongation_smoother']

def jacobi_prolongation_smoother(S, T, omega=4.0/3.0, degree=1):
    """Jacobi prolongation smoother
   
    Parameters
    ----------
    S : {csr_matrix, bsr_matrix}
        Sparse NxN matrix used for smoothing.  Typically, A or the
        "filtered matrix" obtained from A by lumping weak connections
        onto the diagonal of A.
    T : {csr_matrix, bsr_matrix}
        Tentative prolongator
    omega : {scalar}
        Damping parameter

    Returns
    -------
    P : {csr_matrix, bsr_matrix}
        Smoothed (final) prolongator defined by P = (I - omega/rho(K) K) * T
        where K = diag(S)^-1 * S and rho(K) is an approximation to the 
        spectral radius of K.

    Examples
    --------
    >>> from pyamg.aggregation import jacobi_prolongation_smoother
    >>> from pyamg.gallery import poisson
    >>> from scipy.sparse import coo_matrix
    >>> import numpy
    >>> data = numpy.ones((6,))
    >>> row = numpy.arange(0,6)
    >>> col = numpy.kron([0,1],numpy.ones((3,)))
    >>> T = coo_matrix((data,(row,col)),shape=(6,2)).tocsr()
    >>> T.todense()
    matrix([[ 1.,  0.],
            [ 1.,  0.],
            [ 1.,  0.],
            [ 0.,  1.],
            [ 0.,  1.],
            [ 0.,  1.]])
    >>> A = poisson((6,),format='csr')
    >>> P = jacobi_prolongation_smoother(A,T)
    >>> P.todense()
    matrix([[ 0.64930164,  0.        ],
            [ 1.        ,  0.        ],
            [ 0.64930164,  0.35069836],
            [ 0.35069836,  0.64930164],
            [ 0.        ,  1.        ],
            [ 0.        ,  0.64930164]])

    """

    D = S.diagonal()
    D_inv = 1.0 / D
    D_inv[D == 0] = 0

    D_inv_S = scale_rows(S, D_inv, copy=True)
    D_inv_S *= omega/approximate_spectral_radius(D_inv_S)

    P = T
    for i in range(degree):
        P = P - (D_inv_S*P)

    return P


def richardson_prolongation_smoother(S, T, omega=4.0/3.0, degree=1):
    """Richardson prolongation smoother
   
    Parameters
    ----------
    S : {csr_matrix, bsr_matrix}
        Sparse NxN matrix used for smoothing.  Typically, A or the
        "filtered matrix" obtained from A by lumping weak connections
        onto the diagonal of A.
    T : {csr_matrix, bsr_matrix}
        Tentative prolongator
    omega : {scalar}
        Damping parameter

    Returns
    -------
    P : {csr_matrix, bsr_matrix}
        Smoothed (final) prolongator defined by P = (I - omega/rho(S) S) * T
        where rho(S) is an approximation to the spectral radius of S.

    Examples
    --------
    >>> from pyamg.aggregation import richardson_prolongation_smoother
    >>> from pyamg.gallery import poisson
    >>> from scipy.sparse import coo_matrix
    >>> import numpy
    >>> data = numpy.ones((6,))
    >>> row = numpy.arange(0,6)
    >>> col = numpy.kron([0,1],numpy.ones((3,)))
    >>> T = coo_matrix((data,(row,col)),shape=(6,2)).tocsr()
    >>> T.todense()
    matrix([[ 1.,  0.],
            [ 1.,  0.],
            [ 1.,  0.],
            [ 0.,  1.],
            [ 0.,  1.],
            [ 0.,  1.]])
    >>> A = poisson((6,),format='csr')
    >>> P = richardson_prolongation_smoother(A,T)
    >>> P.todense()
    matrix([[ 0.64930164,  0.        ],
            [ 1.        ,  0.        ],
            [ 0.64930164,  0.35069836],
            [ 0.35069836,  0.64930164],
            [ 0.        ,  1.        ],
            [ 0.        ,  0.64930164]])

    """

    weight = omega/approximate_spectral_radius(S)

    P = T
    for i in range(degree):
        P = P - weight*(S*P)

    return P

def kaczmarz_jacobi_prolongation_smoother(S, T, omega=4.0/3.0, degree=1):
    """Jacobi prolongation smoother for the normal equations (i.e. Kaczmarz)
   
    Parameters
    ----------
    S : {csr_matrix, bsr_matrix}
        Sparse NxN matrix used for smoothing.  Typically, A or the
        "filtered matrix" obtained from A by lumping weak connections
        onto the diagonal of A.
    T : {csr_matrix, bsr_matrix}
        Tentative prolongator
    omega : {scalar}
        Damping parameter

    Returns
    -------
    P : {csr_matrix, bsr_matrix}
        Smoothed (final) prolongator

    Examples
    --------
    >>> from pyamg.aggregation import kaczmarz_jacobi_prolongation_smoother
    >>> from pyamg.gallery import poisson
    >>> from scipy.sparse import coo_matrix
    >>> import numpy
    >>> data = numpy.ones((6,))
    >>> row = numpy.arange(0,6)
    >>> col = numpy.kron([0,1],numpy.ones((3,)))
    >>> T = coo_matrix((data,(row,col)),shape=(6,2)).tocsr()
    >>> T.todense()
    matrix([[ 1.,  0.],
            [ 1.,  0.],
            [ 1.,  0.],
            [ 0.,  1.],
            [ 0.,  1.],
            [ 0.,  1.]])
    >>> A = poisson((6,),format='csr')
    >>> P = kaczmarz_jacobi_prolongation_smoother(A,T)
    >>> P.todense()
    matrix([[ 0.78365913,  0.        ],
            [ 1.19831246, -0.09014203],
            [ 0.72957391,  0.27042609],
            [ 0.27042609,  0.72957391],
            [-0.09014203,  1.19831246],
            [ 0.        ,  0.78365913]])

    """

    # Form Dinv for S*S.H
    D_inv = get_diagonal(S, norm_eq=2, inv=True)
    D_inv_S = scale_rows(S, D_inv, copy=True)

    # Approximate Spectral radius by defining a matvec for S.T*D_inv_S
    ST = S.conjugate().T.asformat(D_inv_S.format)
    class matvec_mat:
    
        def __init__(self, matvec, shape, dtype):
            self.shape = shape
            self.matvec = matvec
            self.__mul__ = matvec
            self.dtype = dtype
    
    def matmul(A,B,x):
        return A*(B*x)
    
    StDS_matmul = lambda x:matmul(ST, D_inv_S, x)
    StDS = matvec_mat(StDS_matmul, S.shape, S.dtype)
    omega = omega/approximate_spectral_radius(StDS)

    P = T
    for i in range(degree):
        P = P - omega*(ST*(D_inv_S*P))

    return P

def kaczmarz_richardson_prolongation_smoother(S, T, omega=4.0/3.0, degree=1):
    """Richardson prolongation smoother for the normal equations (i.e. Kaczmarz)
   
    Parameters
    ----------
    S : {csr_matrix, bsr_matrix}
        Sparse NxN matrix used for smoothing.  Typically, A or the
        "filtered matrix" obtained from A by lumping weak connections
        onto the diagonal of A.
    T : {csr_matrix, bsr_matrix}
        Tentative prolongator
    omega : {scalar}
        Damping parameter

    Returns
    -------
    P : {csr_matrix, bsr_matrix}
        Smoothed (final) prolongator 

    Examples
    --------
    >>> from pyamg.aggregation import kaczmarz_richardson_prolongation_smoother
    >>> from pyamg.gallery import poisson
    >>> from scipy.sparse import coo_matrix
    >>> import numpy
    >>> data = numpy.ones((6,))
    >>> row = numpy.arange(0,6)
    >>> col = numpy.kron([0,1],numpy.ones((3,)))
    >>> T = coo_matrix((data,(row,col)),shape=(6,2)).tocsr()
    >>> T.todense()
    matrix([[ 1.,  0.],
            [ 1.,  0.],
            [ 1.,  0.],
            [ 0.,  1.],
            [ 0.,  1.],
            [ 0.,  1.]])
    >>> A = poisson((6,),format='csr')
    >>> P = kaczmarz_richardson_prolongation_smoother(A,T)
    >>> P.todense()
    matrix([[ 0.81551599,  0.        ],
            [ 1.18448401, -0.09224201],
            [ 0.72327398,  0.27672602],
            [ 0.27672602,  0.72327398],
            [-0.09224201,  1.18448401],
            [ 0.        ,  0.81551599]])

    """

    # Approximate Spectral radius by defining a matvec for S*S.H
    ST = S.conjugate().T.asformat(S.format)
    class matvec_mat:
    
        def __init__(self, matvec, shape, dtype):
            self.shape = shape
            self.matvec = matvec
            self.__mul__ = matvec
            self.dtype = dtype
    
    def matmul(A,B,x):
        return A*(B*x)
    
    StS_matmul = lambda x:matmul(ST, S, x)
    StS = matvec_mat(StS_matmul, S.shape, S.dtype)
    omega = omega/approximate_spectral_radius(StS)

    P = T
    for i in range(degree):
        P = P - omega*(ST*(S*P))

    return P


""" sa_energy_min + helper functions minimize the energy of a tentative prolongator for use in SA """


########################################################################################################
#   Helper function for the energy minimization prolongator generation routine

def Satisfy_Constraints(U, B, BtBinv):
    """U is the prolongator update.
       Project out components of U such that U*B = 0

    Parameters
    ----------
    U : {bsr_matrix}
        m x n sparse bsr matrix
        Update to the prolongator
    B : {array}
        n x k array of the coarse grid near nullspace vectors
    BtBinv : {array}
        Local inv(B_i.H*B_i) matrices for each supernode, i 
        B_i is B restricted to the sparsity pattern of supernode i in U

    Returns
    -------
    Updated U, so that U*B = 0.  
    Update is computed by orthogonally (in 2-norm) projecting 
    out the components of span(B) in U in a row-wise fashion.

    See Also
    --------
    See the function energy_prolongation_smoother in smooth.py 

    """
    
    RowsPerBlock = U.blocksize[0]
    ColsPerBlock = U.blocksize[1]
    num_blocks = U.indices.shape[0]
    num_block_rows = U.shape[0]/RowsPerBlock

    UB = numpy.ravel(U*B)

    # Apply constraints, noting that we need the conjugate of B 
    # for use as Bi.H in local projection
    pyamg.amg_core.satisfy_constraints_helper(RowsPerBlock, ColsPerBlock, 
            num_blocks, num_block_rows, 
            numpy.conjugate(numpy.ravel(B)), UB, numpy.ravel(BtBinv), 
            U.indptr, U.indices, numpy.ravel(U.data))
        
    return U

    

########################################################################################################


def energy_prolongation_smoother(A, T, Atilde, B, SPD=True, maxiter=4, tol=1e-8, degree=1):
    """Minimize the energy of the coarse basis functions (columns of T)

    Parameters
    ----------

    A : {csr_matrix, bsr_matrix}
        Sparse NxN matrix
    T : {bsr_matrix}
        Tentative prolongator, a NxM sparse matrix (M < N)
    Atilde : {csr_matrix}
        Strength of connection matrix
    B : {array}
        Near-nullspace modes for coarse grid.  Has shape (M,k) where
        k is the number of coarse candidate vectors.
    SPD : boolean
        Booolean denoting symmetric (or Hermitian) positive-definiteness of A
    maxiter : integer
        Number of energy minimization steps to apply to the prolongator
    tol : scalar
        Minimization tolerance
   
    Returns
    -------
    P : {bsr_matrix}
        Smoothed prolongator

    Examples
    --------
    >>> from pyamg.aggregation import energy_prolongation_smoother
    >>> from pyamg.gallery import poisson
    >>> from scipy.sparse import coo_matrix
    >>> import numpy
    >>> data = numpy.ones((6,))
    >>> row = numpy.arange(0,6)
    >>> col = numpy.kron([0,1],numpy.ones((3,)))
    >>> T = coo_matrix((data,(row,col)),shape=(6,2)).tocsr()
    >>> print T.todense()
    [[ 1.  0.]
     [ 1.  0.]
     [ 1.  0.]
     [ 0.  1.]
     [ 0.  1.]
     [ 0.  1.]]
    >>> A = poisson((6,),format='csr')
    >>> P = energy_prolongation_smoother(A,T,A,numpy.ones((2,1), dtype=float))
    >>> print P.todense()
    [[ 1.          0.        ]
     [ 1.          0.        ]
     [ 0.66666667  0.33333333]
     [ 0.33333333  0.66666667]
     [ 0.          1.        ]
     [ 0.          1.        ]]

    References
    ----------
    .. [1] Jan Mandel, Marian Brezina, and Petr Vanek
       "Energy Optimization of Algebraic Multigrid Bases"
       Computing 62, 205-228, 1999
       http://dx.doi.org/10.1007/s006070050022
    
    """
    
    #====================================================================
    #Test Inputs
    if maxiter < 0:
        raise ValueError('maxiter must be > 0')
    if tol > 1:
        raise ValueError('tol must be <= 1') 
   
    if isspmatrix_csr(A):
        A = A.tobsr(blocksize=(1,1), copy=False)
    elif isspmatrix_bsr(A):
        pass
    else:
        raise TypeError("A must be csr_matrix or bsr_matrix")

    if isspmatrix_csr(T):
        T = T.tobsr(blocksize=(1,1), copy=False)
    elif isspmatrix_bsr(T):
        pass
    else:
        raise TypeError("T must be csr_matrix or bsr_matrix")

    if Atilde is None:
        AtildeCopy = csr_matrix( (numpy.ones(len(A.indices)), A.indices.copy(), A.indptr.copy()), shape=(A.shape[0]/A.blocksize[0], A.shape[1]/A.blocksize[1]))
    else:
        AtildeCopy = Atilde.copy()

    if not isspmatrix_csr(Atilde):
        raise TypeError("Atilde must be csr_matrix")

    if T.blocksize[0] != A.blocksize[0]:
        raise ValueError("T's row-blocksize should be the same as A's blocksize")
    
    if B.shape[0] != T.shape[1]:
        raise ValueError("B is the candidates for the coarse grid. \
                            num_rows(b) = num_cols(T)")

    if min(T.nnz, Atilde.nnz, A.nnz) == 0:
        return T
    
    #====================================================================
    # Retrieve problem information
    Nfine = T.shape[0]
    Ncoarse = T.shape[1]
    NullDim = B.shape[1]
    #Number of PDEs per point is defined implicitly by block size
    numPDEs = A.blocksize[0]
    #====================================================================
    
    #====================================================================
    # Expand the allowed sparsity pattern for P through multiplication by Atilde
    T.sort_indices()
    Sparsity_Pattern = csr_matrix( (numpy.ones(T.indices.shape), T.indices, T.indptr), 
                                    shape=(T.shape[0]/T.blocksize[0],T.shape[1]/T.blocksize[1])  )
    AtildeCopy.data[:] = 1.0
    for i in range(degree):
        Sparsity_Pattern = AtildeCopy*Sparsity_Pattern
    
    del AtildeCopy
    #UnAmal returns a BSR matrix
    Sparsity_Pattern = UnAmal(Sparsity_Pattern, T.blocksize[0], T.blocksize[1])
    Sparsity_Pattern.sort_indices()
    #====================================================================

    #====================================================================
    #Construct array of inv(Bi'Bi), where Bi is B restricted to row i's sparsity pattern in 
    #   Sparsity Pattern.  This array is used multiple times in the Satisfy_Constraints routine.

    ColsPerBlock = Sparsity_Pattern.blocksize[1]
    RowsPerBlock = Sparsity_Pattern.blocksize[0]
    Nnodes = Nfine/RowsPerBlock

    BtBinv = numpy.zeros((Nnodes,NullDim,NullDim), dtype=B.dtype) 
    BsqCols = sum(range(NullDim+1))
    Bsq = numpy.zeros((Ncoarse,BsqCols), dtype=B.dtype)
    counter = 0
    for i in range(NullDim):
        for j in range(i,NullDim):
            Bsq[:,counter] = numpy.conjugate(numpy.ravel(numpy.asarray(B[:,i])))*numpy.ravel(numpy.asarray(B[:,j]))
            counter = counter + 1
    
    pyamg.amg_core.calc_BtB(NullDim, Nnodes, ColsPerBlock,
            numpy.ravel(numpy.asarray(Bsq)), 
        BsqCols, numpy.ravel(numpy.asarray(BtBinv)), Sparsity_Pattern.indptr, Sparsity_Pattern.indices)
    # pinv_array inverts each block in BtBinv
    pyamg.amg_core.pinv_array(numpy.ravel(BtBinv), Nnodes, NullDim, 'F')
    #====================================================================
    
    #====================================================================
    #Iteratively minimize the energy of T subject to the constraints of Sparsity_Pattern
    #   and maintaining T's effect on B, i.e. T*B = (T+Update)*B, i.e. Update*B = 0 
    i = 0
    if SPD:
        # Preallocate
        AP = bsr_matrix((numpy.zeros(Sparsity_Pattern.data.shape, dtype=T.dtype), Sparsity_Pattern.indices, Sparsity_Pattern.indptr), 
                         shape=(Sparsity_Pattern.shape) )

        #Apply CG with diagonal preconditioning
        Dinv = get_diagonal(A, norm_eq=False, inv=True)

        # Calculate initial residual
        #   Equivalent to R = -A*T;    R = R.multiply(Sparsity_Pattern)
        #   with the added constraint that R has an explicit 0 wherever 
        #   R is 0 and Sparsity_Pattern is not
        R = bsr_matrix((numpy.zeros(Sparsity_Pattern.data.shape, dtype=T.dtype), Sparsity_Pattern.indices, Sparsity_Pattern.indptr), 
                        shape=(Sparsity_Pattern.shape) )
        # This gives us the same sparsity data structures as T in BSC format.
        # It has the added benefit of TBSC.data looking like T.tobsc().data, but 
        # with each block in data looking like it is in column-major format, 
        # which is needed for the gemm in incomplete_BSRmatmat.
        TBSC = -1.0*T.T.tobsr()
        TBSC.sort_indices()
        A.sort_indices()
        pyamg.amg_core.incomplete_BSRmatmat(A.indptr,    A.indices,
                numpy.ravel(A.data), 
                                                  TBSC.indptr, TBSC.indices,
                                                  numpy.ravel(TBSC.data),
                                                  R.indptr,    R.indices,
                                                  numpy.ravel(R.data),      
                                                  T.shape[0],  T.blocksize[0],T.blocksize[1])
        del TBSC
        
        # Enforce R*B = 0
        Satisfy_Constraints(R, B, BtBinv)
    
        if R.nnz == 0:
            print "Error in sa_energy_min(..).  Initial R no nonzeros on a level.  Calling Default Prolongator Smoother\n"
            return jacobi_prolongation_smoother(Atilde, T)
        
        #Calculate max norm of the residual
        if R.data.shape[0] == 0:
            resid = 0.0
        else:
            resid = abs(R.data).max()
        #print "Energy Minimization of Prolongator --- Iteration 0 --- r = " + str(resid)

        while i < maxiter and resid > tol:
            #Apply diagonal preconditioner
            Z = scale_rows(R, Dinv)
    
            #Frobenius innerproduct of (R,Z) = sum( numpy.conjugate(rk).*zk)
            newsum = (R.conjugate().multiply(Z)).sum()
                
            #P is the search direction, not the prolongator, which is T.    
            if(i == 0):
                P = Z
            else:
                beta = newsum/oldsum
                P = Z + beta*P
            oldsum = newsum
    
            # This gives us the same sparsity data structures as P in BSC format
            # It has the added benefit of TBSC.data looking like P.tobsc().data, but 
            # with each block in data looking like it is in column-major format, 
            # which is needed for the gemm in incomplete_BSRmatmat.
            PBSC = P.T.tobsr()
            PBSC.sort_indices()
            
            # Calculate new direction and enforce constraints
            #   Equivalent to:  AP = A*P;    AP = AP.multiply(Sparsity_Pattern)
            #   with the added constraint that explicit zeros are in AP wherever 
            #   AP = 0 and Sparsity_Pattern does not
            pyamg.amg_core.incomplete_BSRmatmat(A.indptr,    A.indices,
                    numpy.ravel(A.data), 
                                                      PBSC.indptr,
                                                      PBSC.indices,   numpy.ravel(PBSC.data),
                                                      AP.indptr,   AP.indices,
                                                      numpy.ravel(AP.data),      
                                                      T.shape[0],  T.blocksize[0], T.blocksize[1])

            # Enforce AP*B = 0
            Satisfy_Constraints(AP, B, BtBinv)
            
            #Frobenius innerproduct of (P, AP)
            alpha = newsum/(P.conjugate().multiply(AP)).sum()
    
            #Update the prolongator, T
            T = T + alpha*P 
    
            #Update residual
            R = R - alpha*AP
            
            i += 1
            if R.data.shape[0] == 0:
                resid = 0.0
            else:
                resid = abs(R.data).max()
            #print "Energy Minimization of Prolongator --- Iteration " + str(i) + " --- r = " + str(resid)
     
    else:   
        #For non-SPD system, apply CG on Normal Equations with Diagonal Preconditioning (requires transpose)
        Ah = A.H
        Ah.sort_indices()
        
        # Preallocate
        AP = bsr_matrix((numpy.zeros(Sparsity_Pattern.data.shape, dtype=T.dtype), Sparsity_Pattern.indices, Sparsity_Pattern.indptr), 
                          shape=(Sparsity_Pattern.shape) )

        # D for A.H*A
        Dinv = get_diagonal(A, norm_eq=1, inv=True)

        # Calculate initial residual
        #   Equivalent to R = -Ah*(A*T);    R = R.multiply(Sparsity_Pattern)
        #   with the added constraint that R has an explicit 0 wherever 
        #   R is 0 and Sparsity_Pattern is not
        R = bsr_matrix((numpy.zeros(Sparsity_Pattern.data.shape, dtype=T.dtype), Sparsity_Pattern.indices, Sparsity_Pattern.indptr), 
                        shape=(Sparsity_Pattern.shape) )
        # This gives us the same sparsity data structures as T in BSC format
        # It has the added benefit of TBSC.data looking like T.tobsc().data, but 
        # with each block in data looking like it is in column-major format, 
        # which is needed for the gemm in incomplete_BSRmatmat.
        ATBSC = -1.0*(A*T).T.tobsr()
        ATBSC.sort_indices()
        pyamg.amg_core.incomplete_BSRmatmat(Ah.indptr,    Ah.indices,
                numpy.ravel(Ah.data), 
                                                  ATBSC.indptr, ATBSC.indices,
                                                  numpy.ravel(ATBSC.data),
                                                  R.indptr,    R.indices,
                                                  numpy.ravel(R.data),      
                                                  T.shape[0],   T.blocksize[0],T.blocksize[1])
        del ATBSC

        # Enforce R*B = 0
        Satisfy_Constraints(R, B, BtBinv)
    
        if R.nnz == 0:
            print "Error in sa_energy_min(..).  Initial R no nonzeros on a level.  Calling Default Prolongator Smoother\n"
            return jacobi_prolongation_smoother(Atilde, T)
        
        #Calculate max norm of the residual
        if R.data.shape[0] == 0:
            resid = 0.0
        else:
            resid = abs(R.data).max()
        #print "Energy Minimization of Prolongator --- Iteration 0 --- r = " + str(resid)

        while i < maxiter and resid > tol:
            #Apply diagonal preconditioner
            Z = scale_rows(R, Dinv)
    
            #Frobenius innerproduct of (R,Z) = sum(rk.*zk)
            newsum = (R.conjugate().multiply(Z)).sum()
                
            #P is the search direction, not the prolongator, which is T.    
            if(i == 0):
                P = Z
            else:
                beta = newsum/oldsum
                P = Z + beta*P
            oldsum = newsum
    
            # This gives us the same sparsity data structures as AP in BSC format
            # It has the added benefit of AP_BSC.data looking like AP.tobsc().data, but 
            # with each block in data looking like it is in column-major format, 
            # which is needed for the gemm in incomplete_BSRmatmat.
            AP_BSC = (A*P).T.tobsr()
            AP_BSC.sort_indices()
            
            #Calculate new direction
            #  Equivalent to:  AP = Ah*(A*P);    AP = AP.multiply(Sparsity_Pattern)
            #  with the added constraint that explicit zeros are in AP wherever 
            #  AP = 0 and Sparsity_Pattern does not
            pyamg.amg_core.incomplete_BSRmatmat(Ah.indptr,
                    Ah.indices,     numpy.ravel(Ah.data), 
                                                      AP_BSC.indptr,
                                                      AP_BSC.indices, numpy.ravel(AP_BSC.data),
                                                      AP.indptr,
                                                      AP.indices,     numpy.ravel(AP.data),      
                                                      T.shape[0],    T.blocksize[0], T.blocksize[1])
            
            # Enforce AP*B = 0
            Satisfy_Constraints(AP, B, BtBinv)
            
            #Frobenius innerproduct of (P, AP)
            alpha = newsum/(P.conjugate().multiply(AP)).sum()
    
            #Update the prolongator, T
            T = T + alpha*P 
    
            #Update residual
            R = R - alpha*AP
            
            i += 1
            if R.data.shape[0] == 0:
                resid = 0.0
            else:
                resid = abs(R.data).max()
            #print "Energy Minimization of Prolongator --- Iteration " + str(i) + " --- r = " + str(resid)
    
#====================================================================
# Previous non-SPD minimization strategy
#        #Apply min-res to the nonsymmetric system
#        while i < maxiter and resid > tol:
#    
#            #P is the search direction, not the prolongator
#            P = A*R
#    
#            #Enforce constraints on P
#            P = P.multiply(Sparsity_Pattern)
#            if P.nnz < Sparsity_Pattern.nnz:
#                # ugly hack to give P the same sparsity pattern as Sparsity_Pattern
#                P = P + 1e-100*Sparsity_Pattern
#                Pshape = P.data.shape
#                P.data = P.data.reshape(-1,)
#                P.data[P.data == 1e-100] = 0.0
#                P.data = P.data.reshape(Pshape)           
#
#            Satisfy_Constraints(P, B, BtBinv)
#    
#            #Frobenius innerproduct of (P, R)
#            numer = (P.multiply(R)).sum()
#            
#            #Frobenius innerproduct of (P, P)
#            denom = (P.multiply(P)).sum()
#    
#            alpha = numer/denom
#    
#            #Update prolongator
#            T = T + alpha*R
#    
#            #Update residual
#            R = R - alpha*P
#            
#            i += 1
#            resid = max(R.data.flatten().__abs__())
#            #print "Energy Minimization of Prolongator --- Iteration " + str(i) + " --- r = " + str(resid)
#====================================================================
    T.eliminate_zeros()
    return T

