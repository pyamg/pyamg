"""Methods to smooth tentative prolongation operators"""

__docformat__ = "restructuredtext en"

import numpy
import scipy
from scipy.sparse import csr_matrix, isspmatrix_csr, bsr_matrix, isspmatrix_bsr, spdiags
from pyamg.util.utils import scale_rows, get_diagonal, get_block_diag, UnAmal
from pyamg.util.linalg import approximate_spectral_radius, pinv_array
import pyamg.amg_core

__all__ = ['jacobi_prolongation_smoother', 'richardson_prolongation_smoother', 
           'energy_prolongation_smoother']


# Satisfy_Constraints is a helper function for prolongation smoothing routines
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


def jacobi_prolongation_smoother(S, T, C, B, omega=4.0/3.0, degree=1, filter=False, weighting='diagonal'):
    """Jacobi prolongation smoother
   
    Parameters
    ----------
    S : {csr_matrix, bsr_matrix}
        Sparse NxN matrix used for smoothing.  Typically, A.
    T : {csr_matrix, bsr_matrix}
        Tentative prolongator
    C : {csr_matrix, bsr_matrix}
        Strength-of-connection matrix
    B : {array}
        Near nullspace modes for the coarse grid such that T*B 
        exactly reproduces the fine grid near nullspace modes
    omega : {scalar}
        Damping parameter
    filter : {boolean}
        If true, filter S before smoothing T.  This option can greatly control
        complexity.
    weighting : {string}
        'block', 'diagonal' or 'local' weighting for constructing the Jacobi D
        'local': Uses a local row-wise weight based on the Gershgorin estimate.
          Avoids any potential under-damping due to inaccurate spectral radius
          estimates.
        'block': If A is a BSR matrix, use a block diagonal inverse of A  
        'diagonal': Classic Jacobi D = diagonal(A)

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
    >>> P = jacobi_prolongation_smoother(A,T,A,numpy.ones((2,1)))
    >>> P.todense()
    matrix([[ 0.64930164,  0.        ],
            [ 1.        ,  0.        ],
            [ 0.64930164,  0.35069836],
            [ 0.35069836,  0.64930164],
            [ 0.        ,  1.        ],
            [ 0.        ,  0.64930164]])

    """

    # preprocess weighting
    if weighting == 'block':
        if isspmatrix_csr(S):
            weighting = 'diagonal'
        elif isspmatrix_bsr(S):
            if S.blocksize[0] == 1:
                weighting = 'diagonal'
    
    if filter:
        ##
        # Implement filtered prolongation smoothing for the general case by
        # utilizing satisfy constraints

        # Retrieve problem information
        Nfine = T.shape[0]
        Ncoarse = T.shape[1]
        NullDim = B.shape[1]
        ColsPerBlock = T.blocksize[1]
        RowsPerBlock = T.blocksize[0]
        Nnodes = Nfine/RowsPerBlock
        if isspmatrix_bsr(S):
            numPDEs = S.blocksize[0]
        else:
            numPDEs = 1

        # Create a filtered S with entries dropped that aren't in C
        C = UnAmal(C, numPDEs, numPDEs)
        S = S.multiply(C)
        S.eliminate_zeros()

    if weighting == 'diagonal':
        # Use diagonal of S
        D_inv = get_diagonal(S, inv=True)
        D_inv_S = scale_rows(S, D_inv, copy=True)
        D_inv_S = (omega/approximate_spectral_radius(D_inv_S))*D_inv_S
    elif weighting == 'block':
        # Use block diagonal of S
        D_inv = get_block_diag(S, blocksize=S.blocksize[0], inv_flag=True)
        D_inv = bsr_matrix( (D_inv, numpy.arange(D_inv.shape[0]), \
                         numpy.arange(D_inv.shape[0]+1)), shape = S.shape)
        D_inv_S = D_inv*S
        D_inv_S = (omega/approximate_spectral_radius(D_inv_S))*D_inv_S
    elif weighting == 'local':
        # Use the Gershgorin estimate as each row's weight, instead of a global
        # spectral radius estimate
        D = numpy.abs(S)*numpy.ones((S.shape[0],1), dtype=S.dtype)
        D_inv = 1.0 / numpy.array(numpy.abs(D), dtype=S.dtype)
        D_inv[D == 0] = 0

        D_inv_S = scale_rows(S, D_inv, copy=True)
        D_inv_S = omega*D_inv_S
    else:
        raise ValueError('Incorrect weighting option')

    
    if filter: 
        ##
        # Carry out Jacobi, but after calculating the prolongator update, U,
        # apply satisfy constraints so that U*B = 0

        ##
        # Carry out Jacobi with a satisfy constraints call
        P = T
        for i in range(degree):
            U =  (D_inv_S*P).tobsr(blocksize=P.blocksize)
            
            ##
            # Enforce U*B = 0 
            # (1) Construct array of inv(Bi'Bi), where Bi is B restricted 
            # to the sparsity pattern of row i. This array is used
            # multiple times in the Satisfy_Constraints routine.
            BtBinv = numpy.zeros((Nnodes,NullDim,NullDim), dtype=B.dtype) 
            BsqCols = sum(range(NullDim+1))
            Bsq = numpy.zeros((Ncoarse,BsqCols), dtype=B.dtype)
            counter = 0
            for i in range(NullDim):
                for j in range(i,NullDim):
                    Bsq[:,counter] = numpy.conjugate(numpy.ravel(numpy.asarray(B[:,i])))*numpy.ravel(numpy.asarray(B[:,j]))
                    counter = counter + 1
            
            pyamg.amg_core.calc_BtB(NullDim, Nnodes, ColsPerBlock,
                    numpy.ravel(numpy.asarray(Bsq)), BsqCols, 
                    numpy.ravel(numpy.asarray(BtBinv)), U.indptr, U.indices)
            
            ## 
            # Invert each block of BtBinv
            if NullDim < 7:
                # This specialized routine lacks robustness for large matrices
                # Also, this specialized routine carries out an implicit transpose of BtBinv
                pyamg.amg_core.pinv_array(numpy.ravel(BtBinv), Nnodes, NullDim, 'F')
            else:
                BtBinv = BtBinv.transpose((0,2,1)).copy()
                pinv_array(BtBinv)

            # (2) Apply satisfy constraints
            Satisfy_Constraints(U, B, BtBinv)
            
            # Update P
            P = P - U

    else:
        ##
        # Carry out Jacobi as normal
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


""" 
  sa_energy_min + helper functions minimize the energy of a tentative prolongator for use in SA 
"""

def cg_prolongation_smoothing(A, T, B, BtBinv, Sparsity_Pattern, maxiter, tol, weighting='local'):
    '''
    Helper function for energy_prolongation_smoother(...)   

    Use CG to smooth T by solving A T = 0, subject to nullspace
    and sparsity constraints.

    Parameters
    ----------

    A : {csr_matrix, bsr_matrix}
        SPD sparse NxN matrix
    T : {bsr_matrix}
        Tentative prolongator, a NxM sparse matrix (M < N).
        This is initial guess for the equation A T = 0.
        Assumed that T B_c = B_f
    B : {array}
        Near-nullspace modes for coarse grid, i.e., B_c.  
        Has shape (M,k) where k is the number of coarse candidate vectors.
    BtBinv : {array}
        3 dimensional array such that,
        BtBinv[i] = pinv(B_i.H Bi), and B_i is B restricted 
        to the neighborhood (in the matrix graph) of dof of i.
    Sparsity_Pattern : {csr_matrix, bsr_matrix}
        Sparse NxN matrix
        This is the sparsity pattern constraint to enforce on the 
        eventual prolongator
    maxiter : int
        maximum number of iterations
    tol : float
        residual tolerance for A T = 0
    weighting : {string}
        'block', 'diagonal' or 'local' construction of the diagonal preconditioning

    Returns
    -------
    T : {bsr_matrix}
        Smoothed prolongator using conjugate gradients to solve A T = 0, 
        subject to the constraints, T B_c = B_f, and T has no nonzero 
        outside of the sparsity pattern in Sparsity_Pattern.
    See Also
    --------
    See the function energy_prolongation_smoother in smooth.py 

    '''

    # Preallocate
    AP = bsr_matrix((numpy.zeros(Sparsity_Pattern.data.shape, dtype=T.dtype), Sparsity_Pattern.indices, Sparsity_Pattern.indptr), 
                     shape=(Sparsity_Pattern.shape) )

    # CG will be run with diagonal preconditioning
    if weighting == 'diagonal':
        Dinv = get_diagonal(A, norm_eq=False, inv=True)
    elif weighting == 'block':
        Dinv = get_block_diag(A, blocksize=A.blocksize[0], inv_flag=True)
        Dinv = bsr_matrix( (Dinv, numpy.arange(Dinv.shape[0]), numpy.arange(Dinv.shape[0]+1)), shape = A.shape)
    elif weighting == 'local':
        # Based on Gershgorin estimate
        D = numpy.abs(A)*numpy.ones((A.shape[0],1), dtype=A.dtype)
        Dinv = 1.0 / numpy.abs(D)
        Dinv[D == 0] = 0
    else:
        raise ValueError('weighting value is invalid')

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
        print "Error in sa_energy_min(..).  Initial R no nonzeros on a level.  Returning tentative prolongator\n"
        return T
    
    #Calculate Frobenius norm of the residual
    resid = numpy.sqrt((R.data.conjugate()*R.data).sum())
    #print "Energy Minimization of Prolongator --- Iteration 0 --- r = " + str(resid)
    
    i = 0
    while i < maxiter and resid > tol:
        #Apply diagonal preconditioner
        if weighting == 'local' or weighting == 'diagonal':
            Z = scale_rows(R, Dinv)
        else:
            Z = Dinv*R

        #Frobenius inner-product of (R,Z) = sum( numpy.conjugate(rk).*zk)
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
        
        #Frobenius inner-product of (P, AP)
        alpha = newsum/(P.conjugate().multiply(AP)).sum()

        #Update the prolongator, T
        T = T + alpha*P 

        #Update residual
        R = R - alpha*AP
        
        i += 1

        #Calculate Frobenius norm of the residual
        resid = numpy.sqrt((R.data.conjugate()*R.data).sum())
        #print "Energy Minimization of Prolongator --- Iteration " + str(i) + " --- r = " + str(resid)
   
    return T


def cgnr_prolongation_smoothing(A, T, B, BtBinv, Sparsity_Pattern, maxiter, tol, weighting='local'):
    '''
    Helper function for energy_prolongation_smoother(...)   

    Use CGNR to smooth T by solving A T = 0, subject to nullspace
    and sparsity constraints.

    Parameters
    ----------

    A : {csr_matrix, bsr_matrix}
        SPD sparse NxN matrix
        Should be at least nonsymmetric or indefinite
    T : {bsr_matrix}
        Tentative prolongator, a NxM sparse matrix (M < N).
        This is initial guess for the equation A T = 0.
        Assumed that T B_c = B_f
    B : {array}
        Near-nullspace modes for coarse grid, i.e., B_c.  
        Has shape (M,k) where k is the number of coarse candidate vectors.
    BtBinv : {array}
        3 dimensional array such that,
        BtBinv[i] = pinv(B_i.H Bi), and B_i is B restricted 
        to the neighborhood (in the matrix graph) of dof of i.
    Sparsity_Pattern : {csr_matrix, bsr_matrix}
        Sparse NxN matrix
        This is the sparsity pattern constraint to enforce on the 
        eventual prolongator
    maxiter : int
        maximum number of iterations
    tol : float
        residual tolerance for A T = 0
    weighting : {string}
        'block', 'diagonal' or 'local' construction of the diagonal preconditioning
        IGNORED here, only 'diagonal' preconditioning is used.

    Returns
    -------
    T : {bsr_matrix}
        Smoothed prolongator using CGNR to solve A T = 0, 
        subject to the constraints, T B_c = B_f, and T has no nonzero 
        outside of the sparsity pattern in Sparsity_Pattern.
    See Also
    --------
    See the function energy_prolongation_smoother in smooth.py 

    '''
    
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
    
    if True:
        ATBSC = -1.0*(A*T).T.tobsr()
    else:
        # Mimic -(A^* Q A) T 
        # by inserting a multiply by Q after A
        ATBSC = (A*T)
        Sparsity_Pattern.data[:] = 1.0
        ATBSC = ATBSC + 1e-100*Sparsity_Pattern
        ATBSC = ATBSC.multiply(Sparsity_Pattern)
        ATBSC.eliminate_zeros()
        ATBSC.sort_indices()
        Satisfy_Constraints(ATBSC, B, BtBinv)
        ATBSC = -1.0*ATBSC.T.tobsr()
    
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
        print "Error in sa_energy_min(..).  Initial R no nonzeros on a level.  Returning tentative prolongator\n"
        return T
    
    #Calculate Frobenius norm of the residual
    resid = numpy.sqrt((R.data.conjugate()*R.data).sum())
    #print "Energy Minimization of Prolongator --- Iteration 0 --- r = " + str(resid)

    i = 0
    while i < maxiter and resid > tol:
        
        vect = numpy.ravel((A*T).data)
        #print "Iteration " + str(i) + "   Energy = %1.3e"%numpy.sqrt( (vect.conjugate()*vect).sum() )

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
        if True:
            AP_BSC = (A*P).T.tobsr()
            AP_BSC.sort_indices()
        else:    
            # Mimic -(A^* Q A) P 
            # by inserting a multiply by Q after A
            AP_BSC = (A*P)
            Sparsity_Pattern.data[:] = 1.0
            AP_BSC = AP_BSC + 1e-100*Sparsity_Pattern
            AP_BSC = AP_BSC.multiply(Sparsity_Pattern)
            AP_BSC.eliminate_zeros()
            AP_BSC.sort_indices()
            Satisfy_Constraints(AP_BSC, B, BtBinv)
            AP_BSC = AP_BSC.T.tobsr()
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
        
        #Frobenius inner-product of (P, AP)
        alpha = newsum/(P.conjugate().multiply(AP)).sum()
 
        #Update the prolongator, T
        T = T + alpha*P 
 
        #Update residual
        R = R - alpha*AP
        
        i += 1

        #Calculate Frobenius norm of the residual
        resid = numpy.sqrt((R.data.conjugate()*R.data).sum())
        #print "Energy Minimization of Prolongator --- Iteration " + str(i) + " --- r = " + str(resid)

    vect = numpy.ravel((A*T).data)
    #print "Final Iteration " + str(i) + "   Energy = %1.3e"%numpy.sqrt( (vect.conjugate()*vect).sum() )

    return T


def apply_givens(Q, v, k):
    ''' 
    Apply the first k Givens rotations in Q to v 
    
    Parameters
    ----------
    Q : {list} 
        list of consecutive 2x2 Givens rotations 
    v : {array}
        vector to apply the rotations to
    k : {int}
        number of rotations to apply.

    Returns
    -------
    v is changed in place

    Notes
    -----
    This routine is specialized for GMRES.  It assumes that the first Givens
    rotation is for dofs 0 and 1, the second Givens rotation is for dofs 1 and 2,
    and so on.
    '''

    for j in xrange(k):
        Qloc = Q[j]
        v[j:j+2] = scipy.dot(Qloc, v[j:j+2])


def gmres_prolongation_smoothing(A, T, B, BtBinv, Sparsity_Pattern, maxiter, tol, weighting='local'):
    '''
    Helper function for energy_prolongation_smoother(...).

    Use GMRES to smooth T by solving A T = 0, subject to nullspace
    and sparsity constraints.

    Parameters
    ----------

    A : {csr_matrix, bsr_matrix}
        SPD sparse NxN matrix
        Should be at least nonsymmetric or indefinite
    T : {bsr_matrix}
        Tentative prolongator, a NxM sparse matrix (M < N).
        This is initial guess for the equation A T = 0.
        Assumed that T B_c = B_f
    B : {array}
        Near-nullspace modes for coarse grid, i.e., B_c.  
        Has shape (M,k) where k is the number of coarse candidate vectors.
    BtBinv : {array}
        3 dimensional array such that,
        BtBinv[i] = pinv(B_i.H Bi), and B_i is B restricted 
        to the neighborhood (in the matrix graph) of dof of i.
    Sparsity_Pattern : {csr_matrix, bsr_matrix}
        Sparse NxN matrix
        This is the sparsity pattern constraint to enforce on the 
        eventual prolongator
    maxiter : int
        maximum number of iterations
    tol : float
        residual tolerance for A T = 0
    weighting : {string}
        'block', 'diagonal' or 'local' construction of the diagonal preconditioning

    Returns
    -------
    T : {bsr_matrix}
        Smoothed prolongator using GMRES to solve A T = 0, 
        subject to the constraints, T B_c = B_f, and T has no nonzero 
        outside of the sparsity pattern in Sparsity_Pattern.
    See Also
    --------
    See the function energy_prolongation_smoother in smooth.py 

    '''
        
    #For non-SPD system, apply GMRES with Diagonal Preconditioning
    
    # Preallocate space for new search directions
    AV = bsr_matrix((numpy.zeros(Sparsity_Pattern.data.shape, dtype=T.dtype), Sparsity_Pattern.indices, Sparsity_Pattern.indptr), shape=(Sparsity_Pattern.shape) )

    # Preallocate for Givens Rotations, Hessenberg matrix and Krylov Space
    xtype = scipy.sparse.sputils.upcast(A.dtype, T.dtype, B.dtype)
    Q = []                                                 # Givens Rotations
    V = []                                                 # Krylov Space
    vs = []                                                # vs store the pointers to each column of V.
                                                           #   This saves a considerable amount of time.
    H = numpy.zeros( (maxiter+1, maxiter+1), dtype=xtype)  # Upper Hessenberg matrix, which is then 
                                                           #   converted to upper tri with Givens Rots 


    # GMRES will be run with diagonal preconditioning
    if weighting == 'diagonal':
        Dinv = get_diagonal(A, norm_eq=False, inv=True)
    elif weighting == 'block':
        Dinv = get_block_diag(A, blocksize=A.blocksize[0], inv_flag=True)
        Dinv = bsr_matrix( (Dinv, numpy.arange(Dinv.shape[0]), numpy.arange(Dinv.shape[0]+1)), shape = A.shape)
    elif weighting == 'local':
        # Based on Gershgorin estimate
        D = numpy.abs(A)*numpy.ones((A.shape[0],1), dtype=A.dtype)
        Dinv = 1.0 / numpy.abs(D)
        Dinv[D == 0] = 0
    else:
        raise ValueError('weighting value is invalid')

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
    
    #Apply diagonal preconditioner
    if weighting == 'local' or weighting == 'diagonal':
        R = scale_rows(R, Dinv)
    else:
        R = Dinv*R
    
    # Enforce R*B = 0
    Satisfy_Constraints(R, B, BtBinv)

    if R.nnz == 0:
        print "Error in sa_energy_min(..).  Initial R no nonzeros on a level.  Returning tentative prolongator\n"
        return T
    
    # This is the RHS vector for the problem in the Krylov Space
    normr = numpy.sqrt((R.data.conjugate()*R.data).sum())
    g = numpy.zeros((maxiter+1,), dtype=xtype) 
    g[0] = normr
    
    # First Krylov vector
    # V[0] = r/normr
    if normr > 0.0:
        V.append((1.0/normr)*R)

    #print "Energy Minimization of Prolongator --- Iteration 0 --- r = " + str(normr)
    i = -1
    #vect = numpy.ravel((A*T).data)
    #print "Iteration " + str(i+1) + "   Energy = %1.3e"%numpy.sqrt( (vect.conjugate()*vect).sum() )
    #print "Iteration " + str(i+1) + "   Normr  %1.3e"%normr
    while i < maxiter-1 and normr > tol:
        i = i+1

        # Calculate new search direction
        # This gives us the same sparsity data structures as P in BSC format
        # It has the added benefit of TBSC.data looking like P.tobsc().data, but 
        # with each block in data looking like it is in column-major format, 
        # which is needed for the gemm in incomplete_BSRmatmat.
        VBSC = V[i].T.tobsr()
        VBSC.sort_indices()
        #   Equivalent to:  AV = A*V;    AV = AV.multiply(Sparsity_Pattern)
        #   with the added constraint that explicit zeros are in AP wherever 
        #   AP = 0 and Sparsity_Pattern does not
        pyamg.amg_core.incomplete_BSRmatmat(A.indptr, A.indices, numpy.ravel(A.data), 
                                            VBSC.indptr, VBSC.indices, numpy.ravel(VBSC.data),
                                            AV.indptr,   AV.indices,
                                            numpy.ravel(AV.data),      
                                            T.shape[0],  T.blocksize[0], T.blocksize[1])
        
        if weighting == 'local' or weighting == 'diagonal':
            AV = scale_rows(AV, Dinv)
        else:
            AV = Dinv*AV
        
        # Enforce AV*B = 0
        Satisfy_Constraints(AV, B, BtBinv)
        V.append(AV.copy())

        # Modified Gram-Schmidt
        for j in xrange(i+1):
            # Frobenius inner-product
            H[j,i] = (V[j].conjugate().multiply(V[i+1])).sum()
            V[i+1] = V[i+1] - H[j,i]*V[j]

        # Frobenius Norm
        H[i+1,i] = numpy.sqrt( (V[i+1].data.conjugate()*V[i+1].data).sum() )
        
        # Check for breakdown
        if H[i+1,i] != 0.0:
            V[i+1] = (1.0/H[i+1,i])*V[i+1]

        # Apply previous Givens rotations to H
        if i > 0:
            apply_givens(Q, H[:,i], i)
            
        # Calculate and apply next complex-valued Givens Rotation
        if H[i+1, i] != 0:
            h1 = H[i, i]; 
            h2 = H[i+1, i];
            h1_mag = numpy.abs(h1)
            h2_mag = numpy.abs(h2)
            if h1_mag < h2_mag:
                mu = h1/h2
                tau = numpy.conjugate(mu)/numpy.abs(mu)
            else:    
                mu = h2/h1
                tau = mu/numpy.abs(mu)

            denom = numpy.sqrt( h1_mag**2 + h2_mag**2 )               
            c = h1_mag/denom
            s = h2_mag*tau/denom; 
            Qblock = numpy.array([[c, numpy.conjugate(s)], [-s, c]], dtype=xtype)
            Q.append(Qblock)
            
            # Apply Givens Rotation to g, 
            #   the RHS for the linear system in the Krylov Subspace.
            g[i:i+2] = scipy.dot(Qblock, g[i:i+2])
            
            # Apply effect of Givens Rotation to H
            H[i,     i] = scipy.dot(Qblock[0,:], H[i:i+2, i]) 
            H[i+1, i] = 0.0
            
        normr = numpy.abs(g[i+1])
        #print "Iteration " + str(i+1) + "   Normr  %1.3e"%normr

    # End while loop
    

    # Find best update to x in Krylov Space, V.  Solve (i x i) system.
    if i != -1:
        y = scipy.linalg.solve(H[0:i+1,0:i+1], g[0:i+1])
        for j in range(i+1):
            T = T + y[j]*V[j]
    
    #vect = numpy.ravel((A*T).data)
    #print "Final Iteration " + str(i) + "   Energy = %1.3e"%numpy.sqrt( (vect.conjugate()*vect).sum() )

    return T


def energy_prolongation_smoother(A, T, Atilde, B, krylov='cg', maxiter=4, tol=1e-8, degree=1, weighting='local', reorthogonalize=False):
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
    krylov : {string}
        'cg' : for SPD systems.  Solve A T = 0 in a constraint space with CG
        'cgnr' : for nonsymmetric and/or indefinite systems.  
                 Solve A T = 0 in a constraint space with CGNR
        'gmres' : for nonsymmetric and/or indefinite systems.  
                 Solve A T = 0 in a constraint space with GMRES
    maxiter : integer
        Number of energy minimization steps to apply to the prolongator
    tol : {scalar}
        Minimization tolerance
    degree : {int}
        Generate sparsity pattern for P based on (Atilde^degree T)
    weighting : {string}
        'block', 'diagonal' or 'local' construction of the diagonal preconditioning
        'local': Uses a local row-wise weight based on the Gershgorin estimate.
          Avoids any potential under-damping due to inaccurate spectral radius
          estimates.
        'block': If A is a BSR matrix, use a block diagonal inverse of A  
        'diagonal': Use inverse of the diagonal of A
    reorthogonalize : {boolean}
        If True, then re-orthogonalize the columns of P after smoothing.
        Each block column (corresponding to an aggregate) is orthogonalized
        locally.

    Returns
    -------
    T : {bsr_matrix}
        Smoothed prolongator
    B : {array}
        Updated near nullspace modes for coarse grid.  B is updated
        following a local QR done to smoothed prolongator so that
        T*B exactly reproduces the fine grid near nullspace modes

    Notes
    -----
    Only 'diagonal' weighting is supported for the CGNR method, because
    we are working with A^* A and not A.

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
    >>> [P,B] = energy_prolongation_smoother(A,T,A,numpy.ones((2,1), dtype=float))
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

    if not isspmatrix_csr(AtildeCopy):
        raise TypeError("Atilde must be csr_matrix")

    if T.blocksize[0] != A.blocksize[0]:
        raise ValueError("T's row-blocksize should be the same as A's blocksize")
    
    if B.shape[0] != T.shape[1]:
        raise ValueError("B is the candidates for the coarse grid. \
                            num_rows(b) = num_cols(T)")

    if min(T.nnz, AtildeCopy.nnz, A.nnz) == 0:
        return T, B
    
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
    
    pyamg.amg_core.calc_BtB(NullDim, Nnodes, ColsPerBlock, numpy.ravel(numpy.asarray(Bsq)), 
    BsqCols, numpy.ravel(numpy.asarray(BtBinv)), Sparsity_Pattern.indptr, Sparsity_Pattern.indices)
    
    ## 
    # Invert each block of BtBinv, noting that amg_core.calc_BtB(...) returns
    # values in column-major form, thus necessitating the deep transpose
    #   This is the old call to a specialized routine, but lacks robustness 
    #   pyamg.amg_core.pinv_array(numpy.ravel(BtBinv), Nnodes, NullDim, 'F')
    BtBinv = BtBinv.transpose((0,2,1)).copy()
    pinv_array(BtBinv)
    #====================================================================
    
    #====================================================================
    #Iteratively minimize the energy of T subject to the constraints of Sparsity_Pattern
    #   and maintaining T's effect on B, i.e. T*B = (T+Update)*B, i.e. Update*B = 0 
    if krylov == 'cg':
        T = cg_prolongation_smoothing(A, T, B, BtBinv, Sparsity_Pattern, maxiter, tol, weighting)
    elif krylov == 'cgnr':   
        T = cgnr_prolongation_smoothing(A, T, B, BtBinv, Sparsity_Pattern, maxiter, tol, weighting)
    elif krylov == 'gmres':
        T = gmres_prolongation_smoothing(A, T, B, BtBinv, Sparsity_Pattern, maxiter, tol, weighting)
    
    # Replace each block column in P with the Q from a local QR, also update B with a 
    # block multiply with the new R.
    (nPDE, nullDim) = T.blocksize
    if nullDim > 1 and reorthogonalize:
        # Simulate BSC
        T = T.T.tobsr(blocksize=(nullDim,nPDE))

        # Orthogonalize
        for i in xrange(T.shape[0]/nullDim):
            Ai = T.data[T.indptr[i]:T.indptr[i+1], :, :]
            [Q,R] = scipy.linalg.qr(numpy.transpose(Ai, (0,2,1)).reshape(-1,nullDim), 
                                    econ=True, overwrite_a=True)
            # Update B
            B[i*nullDim:(i+1)*nullDim,:] = numpy.dot(R, B[i*nullDim:(i+1)*nullDim,:])
            # Update T
            T.data[T.indptr[i]:T.indptr[i+1], :, :] = numpy.transpose(Q.reshape(T.indptr[i+1]-T.indptr[i], nPDE, nullDim), (0,2,1))

        T = T.T.tobsr(blocksize=(nPDE,nullDim))
    
    T.eliminate_zeros()
    return T,B

