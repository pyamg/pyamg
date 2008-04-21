"""Strength of Connection functions"""

__docformat__ = "restructuredtext en"

from numpy import ones, empty_like, diff

from scipy.sparse import csr_matrix, isspmatrix_csr, isspmatrix_bsr
import multigridtools

__all__ = ['classical_strength_of_connection', 'symmetric_strength_of_connection',
        'ode_strength_of_connection']


#TODO improve docstrings

def classical_strength_of_connection(A,theta):
    """Return a strength of connection matrix using the classical AMG measure

    An off-diagonal entry A[i.j] is a strong connection iff
        -A[i,j] >= theta * max( -A[i,k] )   where k != i
    """
    if not isspmatrix_csr(A): raise TypeError('expected csr_matrix')

    Sp = empty_like(A.indptr)
    Sj = empty_like(A.indices)
    Sx = empty_like(A.data)

    fn = multigridtools.classical_strength_of_connection
    fn(A.shape[0], theta, A.indptr, A.indices, A.data, Sp, Sj, Sx)

    return csr_matrix((Sx,Sj,Sp), shape=A.shape)


def symmetric_strength_of_connection(A, theta=0):
    """Compute a strength of connection matrix using the standard symmetric measure
    
    An off-diagonal connection A[i,j] is strong iff
        abs(A[i,j]) >= theta * sqrt( abs(A[i,i] * A[j,j]) )

    References
    ----------
        Vanek, P. and Mandel, J. and Brezina, M., 
        "Algebraic Multigrid by Smoothed Aggregation for 
        Second and Fourth Order Elliptic Problems", 
        Computing, vol. 56, no. 3, pp. 179--196, 1996.

    """
    #TODO describe case of blocks

    if isspmatrix_csr(A):
        #if theta == 0:
        #    return A
        
        Sp = empty_like(A.indptr)
        Sj = empty_like(A.indices)
        Sx = empty_like(A.data)

        fn = multigridtools.symmetric_strength_of_connection
        fn(A.shape[0], theta, A.indptr, A.indices, A.data, Sp, Sj, Sx)
        
        return csr_matrix((Sx,Sj,Sp),A.shape)

    elif isspmatrix_bsr(A):
        M,N = A.shape
        R,C = A.blocksize

        if R != C:
            raise ValueError('matrix must have square blocks')

        if theta == 0:
            data = ones( len(A.indices), dtype=A.dtype )
            return csr_matrix((data,A.indices,A.indptr),shape=(M/R,N/C))
        else:
            # the strength of connection matrix is based on the 
            # Frobenius norms of the blocks
            data = (A.data*A.data).reshape(-1,R*C).sum(axis=1) 
            A = csr_matrix((data,A.indices,A.indptr),shape=(M/R,N/C))
            return symmetric_strength_of_connection(A, theta)
    else:
        raise TypeError('expected csr_matrix or bsr_matrix') 



from numpy import array, zeros, mat, eye, ones, setdiff1d, min, ravel, diff, mod, repeat, inf, asarray
from scipy.sparse import csr_matrix, isspmatrix_csr, bsr_matrix, isspmatrix_bsr, spdiags
import scipy.sparse
from scipy.linalg import pinv2
from pyamg.utils import approximate_spectral_radius, scale_rows

def ode_strength_of_connection(A, B, epsilon=4.0, k=2, proj_type="l2"):
    """Construct an AMG strength of connection matrix using an ODE based inspiration.

    Parameters
    ----------
    A : {csr_matrix, bsr_matrix}
        Sparse NxN matrix
    B : {array_like}
        Near-nullspace vector(s) stored in NxK array
    epsilon : scalar
        Drop tolerance
    k : integer
        ODE num time steps, step size is assumed to be 1/rho(DinvA)
    proj_type : ['l2','D_A']
        Define norm for constrained min prob, i.e. define projection
   
    Returns
    -------
    Atilde : {csr_matrix}
        Sparse matrix of strength values


    Notes
    -----


    Examples
    --------

    
    References
    ----------

        Jacob Schroder and his homeys
        "Put a title here"


    """

    #Regarding the efficiency TODO listings below, the bulk of the routine's time
    #   is spent inside the main loop that solves the constrained min problem

    #====================================================================
    #Check inputs
    if epsilon < 1.0:
        raise ValueError("expected epsilon > 1.0")
    if k <= 0:
        raise ValueError("number of time steps must be > 0")
    if proj_type not in ['l2', 'D_A']:
        raise VaueError("proj_type must be 'l2' or 'D_A'")
       
    #B must be in mat format, this isn't a deep copy...so OK
    Bmat = mat(B)

    #Amat must be devoid of 0's and have sorted indices
    A.sort_indices()
    A.eliminate_zeros()

    #====================================================================
    # Handle preliminaries for the algorithm
    
    dimen = A.shape[1]
    NullDim = Bmat.shape[1]
    csrflag = isspmatrix_csr(A)
    if (not csrflag) and (isspmatrix_bsr(A) == False):
        raise TypeError("expected csr_matrix or bsr_matrix")
    
    #number of PDEs per point is defined implicitly by block size
    if csrflag:
        numPDEs = 1
    else:
        numPDEs = A.blocksize[0]
    
    #Get spectral radius of Dinv*A, this is the time step size for the ODE 
    D = A.diagonal();
    if (D == 0).any():
        zero_rows = (D == 0).nonzero()[0]
        if (diff(A.tocsr().indptr)[zero_rows] > 0).any():
            pass
            #raise ValueError('zero on diag(A) for nonzero row of A')
        # Zeros on D represent 0 rows, so we can just set D to 1.0 at those locations and then Dinv*A 
        #   at the zero rows of A will still be zero
        D[zero_rows] = 1.0
    Dinv = 1.0/D
    Dinv_A  = scale_rows(A, Dinv, copy=True)
    rho_DinvA = approximate_spectral_radius(Dinv_A)
    
    #Calculate D_A * B for use in minimization problem
    #   Incur the cost of creating a new CSR mat, Dmat, so that we can multiply 
    #   Dmat*B by calling C routine and avoid looping python
    if proj_type == "D_A":
        Dmat = spdiags( [D], [0], dimen, dimen, format = 'csr')
        DB = Dmat*Bmat
        del Dmat
    else:
        DB = Bmat
    #====================================================================
    
    
    #====================================================================
    # Calculate (Atilde^k)^T in two steps.  
    #
    # We want to later access columns of Atilde^k, hence we calculate (Atilde^k)^T so 
    # that columns will be accessed efficiently w.r.t. the CSR format
    
    # First Step.  Calculate (Atilde^p)^T = (Atilde^T)^p, where p is the largest power of two <= k, 
    p = 2;

    if csrflag:    #Maintain CSR format of A in Atilde
        I = scipy.sparse.eye(dimen, dimen, format="csr")
        Atilde = (I - (1.0/rho_DinvA)*Dinv_A)
        Atilde = Atilde.T.tocsr()
    else:       #Maintain BSR format of A in Atilde
        I = bsr_matrix(scipy.sparse.eye(dimen, dimen, format='bsr'),blocksize=A.blocksize)
        Atilde = (I - (1.0/rho_DinvA)*Dinv_A)
        Atilde = Atilde.T

    while p <= k:
        Atilde = Atilde*Atilde
        p = p*2
    
    #Second Step.  Calculate Atilde^p*Atilde^(k-p)
    p = p/2
    if p < k:
        print "The most efficient time stepping for the ODE Strength Method"\
              " is done in powers of two.\nYou have chosen " + str(k) + " time steps."

        if csrflag:
            JacobiStep = (I - (1.0/rho_DinvA)*Dinv_A).T.tocsr()
        else:
            JacobiStep = (I - (1.0/rho_DinvA)*Dinv_A).T
        while p < k:
            Atilde = Atilde*JacobiStep
            p = p+1
        del JacobiStep
    
    #Check matrix Atilde^k vs. above    
    #Atilde2 = ((I - (t/k)*Dinv_A).T)**k
    #diff = (Atilde2 - Atilde).todense()
    #print "Norm of difference is " + str(norm(diff))
    
    del Dinv, Dinv_A
    
    #---Efficiency--- TODO:  Calculate Atilde^k only at the sparsity of A^T, restricting the nonzero pattern
    #            of A^T so that col i only retains the nonzeros that are of the same PDE as i.
    #            However, this will require specialized C routine.  Perhaps look
    #            at mat-mult routines that first precompute the sparsity pattern for
    #            sparse mat-mat-mult.  This could be the easiest thing to do.
        
    #====================================================================
    # Now that the mat-mat part is done, convert to CSR, as this is much faster 
    #   than dealing with BSR matrices natively
    if not csrflag:
        Atilde = Atilde.tocsr()
        Atilde.eliminate_zeros()
    
    #====================================================================
    #Construct and apply a sparsity mask for Atilde that restricts Atilde^T to the nonzero pattern
    #  of A, with the added constraint that row i of Atilde^T retains only the nonzeros that are also
    #  in the same PDE as i. 

    mask = A.copy()
    if not csrflag:
        mask = mask.tocsr()

    #Only consider strength at dofs from your PDE.  Use mask to enforce this by zeroing out
    #   all entries in Atilde that aren't from your PDE.
    if numPDEs > 1:
        row_length = diff(mask.indptr)
        my_pde = mod(range(dimen), numPDEs)
        my_pde = repeat(my_pde, row_length)
        mask.data[ mod(mask.indices, numPDEs) != my_pde ] = 0.0
        del row_length, my_pde
        mask.eliminate_zeros()

    #Apply mask to Atilde, zeros in mask have already been eliminated at start of routine.
    mask.data[:] = 1.0
    Atilde = Atilde.multiply(mask)
    Atilde.eliminate_zeros()
    del mask

    #====================================================================
    # Calculate strength based on constrained min problem of 
    # min( z - B*x ), such that
    # (B*x)|_i = z|_i, i.e. they are equal at point i
    # z = (I - (t/k) Dinv A)^k delta_i
    #
    # Strength is defined as the relative point-wise approx. error between
    # B*x and z.  We don't use the full z in this problem, only that part of
    # z that is in the sparsity pattern of A.
    # 
    # Can use either the D-norm, and inner product, or l2-norm and inner-prod
    # to solve the constrained min problem.  Using D gives scale invariance.
    #
    # This is a quadratic minimization problem with a linear constraint, so
    # we can build a linear system and solve it to find the critical point,
    # i.e. minimum.
    #
    # We exploit a known shortcut for the case of NullDim = 1.  The shortcut is
    # mathematically equivalent to the longer constrained min. problem

    if NullDim == 1:
        # Use shortcut to solve constrained min problem if B is only a vector
        # Strength(i,j) = | 1 - (z(i)/b(i))/(z(j)/b(j)) |
        # These ratios can be calculated by diagonal row and column scalings
        
        #Create necessary Diagonal matrices
        DAtilde = Atilde.diagonal();
        DAtildeDivB = spdiags( [array(DAtilde)/(array(Bmat).reshape(DAtilde.shape))], [0], dimen, dimen, format = 'csr')
        DiagB = spdiags( [array(Bmat).flatten()], [0], dimen, dimen, format = 'csr')
        
        #Calculate Approximation ratio
        Atilde.data = 1.0/Atilde.data

        Atilde = DAtildeDivB*Atilde
        Atilde = Atilde*DiagB

        #Find negative ratios
        neg_ratios =  (Atilde.data < 0.0)

        #Calculate Approximation error
        Atilde.data = abs( 1.0 - Atilde.data)
        
        #Drop negative ratios
        Atilde.data[neg_ratios] = 0.0

        # Set diagonal to 1.0, as each point is perfectly strongly connected to itself.
        I = scipy.sparse.eye(dimen, dimen, format="csr")
        I.data -= Atilde.diagonal()
        Atilde = Atilde + I

    else:
        # For use in computing local B_i^T*B, precompute the dot-* of 
        #   each column of B with each other column.  We also scale by 2.0 
        #   to account for BDB's eventual use in a constrained minimization problem
        BDBCols = sum(range(NullDim+1))
        BDB = zeros((dimen,BDBCols))
        counter = 0
        for i in range(NullDim):
            for j in range(i,NullDim):
                BDB[:,counter] = 2.0*asarray(B[:,i])*ravel(asarray(DB[:,j]))
                counter = counter + 1        
                
        # Use constrained min problem to define strength
        multigridtools.ode_strength_helper(Atilde.data,        Atilde.indptr,    Atilde.indices, 
                                           Atilde.shape[0],    ravel(asarray(B)), ravel(asarray(DB.T)), 
                                           ravel(asarray(BDB)), BDBCols,           NullDim)
        
    #===================================================================
    
    #Apply drop tolerance
    if epsilon != inf:
        Atilde.eliminate_zeros()
        multigridtools.apply_distance_filter(dimen, epsilon, Atilde.indptr, Atilde.indices, Atilde.data)
    
    Atilde.eliminate_zeros()


    #If converted BSR to CSR, convert back
    if not csrflag:
        Atilde = Atilde.tobsr(blocksize=(numPDEs, numPDEs))

    #If BSR matrix, we return amalgamated matrix, i.e. the sparsity structure of the blocks of Atilde
    if not csrflag :
        #Atilde = csr_matrix((data, row, col), shape=(*,*))
        Atilde = csr_matrix((array([ Atilde.data[i,:,:][Atilde.data[i,:,:].nonzero()].min() for i in range(Atilde.indices.shape[0]) ]), \
                             Atilde.indices, Atilde.indptr), shape=(Atilde.shape[0]/numPDEs, Atilde.shape[1]/numPDEs) )

    return Atilde

