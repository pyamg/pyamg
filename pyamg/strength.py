"""Strength of Connection functions"""

__docformat__ = "restructuredtext en"

from numpy import ones, empty_like, diff, conjugate, mat, sqrt, arange, array, real, imag, sign
from warnings import warn
from scipy.sparse import csr_matrix, isspmatrix, isspmatrix_csr, isspmatrix_bsr, SparseEfficiencyWarning, csc_matrix
from scipy.sparse import eye as speye
import multigridtools
from pyamg.util.linalg import approximate_spectral_radius

__all__ = ['classical_strength_of_connection', 'symmetric_strength_of_connection', 'ode_strength_of_connection']

def classical_strength_of_connection(A, theta=0.0):
    """
    Return a strength of connection matrix using the classical AMG measure
    An off-diagonal entry A[i,j] is a strong connection iff::

            | A[i,j] | >= theta * max(| A[i,k] |), where k != i

    Parameters
    ----------
    A : csr_matrix
        Matrix graph defined in sparse format.  Entry A[i,j] describes the
        strength of edge [i,j]
    theta : float
        Threshold parameter in [0,1].

    Returns
    -------
    S : csr_matrix
        Matrix graph defining strong connections.  S[i,j]=1 if vertex i
        is strongly influenced by vertex j.

    See Also
    --------
    symmetric_strength_of_connection : symmetric measure used in SA
    ode_strength_of_connection : relaxation based strength measure

    Notes
    -----
    - A symmetric A does not necessarily yield a symmetric strength matrix S
    - Calls C++ function classical_strength_of_connection
    - The version as implemented is designed form M-matrices.  Trottenberg et
      al. use max A[i,k] over all negative entries, which is the same.  A
      positive edge weight never indicates a strong connection.

    References
    ----------
    .. [1] Briggs, W. L., Henson, V. E., McCormick, S. F., "A multigrid tutorial",
       Second edition. Society for Industrial and Applied Mathematics (SIAM),
       Philadelphia, PA, 2000. xii+193 pp. ISBN: 0-89871-462-1

    .. [2] Trottenberg, U., Oosterlee, C. W., Schuller, A., "Multigrid",
       Academic Press, Inc., San Diego, CA, 2001. xvi+631 pp. ISBN: 0-12-701070-X

    Examples
    --------
    >>> from numpy import array
    >>> from pyamg.gallery import stencil_grid
    >>> from pyamg.strength import classical_strength_of_connection
    >>> n=3
    >>> stencil = array([[-1.0,-1.0,-1.0],
    >>>                  [-1.0, 8.0,-1.0],
    >>>                  [-1.0,-1.0,-1.0]])
    >>> A = stencil_grid(stencil, (n,n), format='csr')
    >>> S = classical_strength_of_connection(A, 0.0)

    """
    if not isspmatrix_csr(A): 
        warn("Implicit conversion of A to csr", SparseEfficiencyWarning)
        A = csr_matrix(A)

    if (theta<0 or theta>1):
        raise ValueError('expected theta in [0,1]')

    Sp = empty_like(A.indptr)
    Sj = empty_like(A.indices)
    Sx = empty_like(A.data)

    fn = multigridtools.classical_strength_of_connection
    fn(A.shape[0], theta, A.indptr, A.indices, A.data, Sp, Sj, Sx)

    return csr_matrix((Sx,Sj,Sp), shape=A.shape)


def symmetric_strength_of_connection(A, theta=0):
    """
    Compute a strength of connection matrix using the standard symmetric measure
    
    An off-diagonal connection A[i,j] is strong iff::

        abs(A[i,j]) >= theta * sqrt( abs(A[i,i]) * abs(A[j,j]) )

    Parameters
    ----------
    A : csr_matrix
        Matrix graph defined in sparse format.  Entry A[i,j] describes the
        strength of edge [i,j]
    theta : float
        Threshold parameter (positive).

    Returns
    -------
    S : csr_matrix
        Matrix graph defining strong connections.  S[i,j]=1 if vertex i
        is strongly influenced by vertex j.

    See Also
    --------
    symmetric_strength_of_connection : symmetric measure used in SA
    ode_strength_of_connection : relaxation based strength measure

    Notes
    -----
        - For vector problems, standard strength measures may produce
          undesirable aggregates.  A "block approach" from Vanek et al. is used
          to replace vertex comparisons with block-type comparisons.  A
          connection between nodes i and j in the block case is strong if::

          ||AB[i,j]|| >= theta * sqrt( ||AB[i,i]||*||AB[j,j]|| ) where AB[k,l]

          is the matrix block (degrees of freedom) associated with nodes k and
          l and ||.|| is a matrix norm, such a Frobenius.
        

    References
    ----------
    .. [1] Vanek, P. and Mandel, J. and Brezina, M., 
       "Algebraic Multigrid by Smoothed Aggregation for 
       Second and Fourth Order Elliptic Problems", 
       Computing, vol. 56, no. 3, pp. 179--196, 1996.
       http://citeseer.ist.psu.edu/vanek96algebraic.html

    Examples
    --------
    >>> from numpy import array
    >>> from pyamg.gallery import stencil_grid
    >>> from pyamg.strength import symmetric_strength_of_connection
    >>> n=3
    >>> stencil = array([[-1.0,-1.0,-1.0],
    >>>                  [-1.0, 8.0,-1.0],
    >>>                  [-1.0,-1.0,-1.0]])
    >>> A = stencil_grid(stencil, (n,n), format='csr')
    >>> S = symmetric_strength_of_connection(A, 0.0)
    """

    if (theta<0):
        raise ValueError('expected a positive theta')

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
            data = (conjugate(A.data)*A.data).reshape(-1,R*C).sum(axis=1) 
            A = csr_matrix((data,A.indices,A.indptr),shape=(M/R,N/C))
            return symmetric_strength_of_connection(A, theta)
    else:
        raise TypeError('expected csr_matrix or bsr_matrix') 

def energy_based_strength_of_connection(A, theta=0.0, k=2):
    """
    Compute a strength of connection matrix using an energy-based measure.
       

    Parameters
    ----------
    A : {sparse-matrix}
        matrix from which to generate strength of connection information
    theta : {float}
        Threshold parameter in [0,1]
    k : {int}
        Number of relaxation steps used to generate strength information

    Returns
    -------
    S : {csr_matrix}
        Matrix graph defining strong connections.  The sparsity pattern
        of S matches that of A.  For BSR matrices, S is a reduced strength
        of connection matrix that describes connections between supernodes.

    Notes
    -----
    This method relaxes with weighted-Jacobi in order to approximate the  
    matrix inverse.  A normalized change of energy is then used to define 
    point-wise strength of connection values.  Specifically, let v be the 
    approximation to the i-th column of the inverse, then 

    (S_ij)^2 = <v_j, v_j>_A / <v, v>_A, 
    
    where v_j = v, such that entry j in v has been zeroed out.  As is common,
    larger values imply a stronger connection.
    
    Current implemenation is a very slow pure-python implementation for 
    experimental purposes, only.

    References
    ----------
    .. [1] Brannick, Brezina, MacLachlan, Manteuffel, McCormick. 
       "An Energy-Based AMG Coarsening Strategy",
       Numerical Linear Algebra with Applications, 
       vol. 13, pp. 133-148, 2006.

    Examples
    --------
    >>> from numpy import array
    >>> from pyamg.gallery import stencil_grid
    >>> from pyamg.strength import energy_based_strength_of_connection
    >>> n=3
    >>> stencil = array([[-1.0,-1.0,-1.0],
    >>>                  [-1.0, 8.0,-1.0],
    >>>                  [-1.0,-1.0,-1.0]])
    >>> A = stencil_grid(stencil, (n,n), format='csr')
    >>> S = energy_based_strength_of_connection(A, 0.0)
    """

    if (theta<0):
        raise ValueError('expected a positive theta')
    if not isspmatrix(A):
        raise ValueError('expected sparse matrix')
    if (k < 0):
        raise ValueError('expected positive number of steps')
    if not isinstance(k, int):
        raise ValueError('expected integer')
    
    if isspmatrix_bsr(A):
        bsr_flag = True
        numPDEs = A.blocksize[0]
        if A.blocksize[0] != A.blocksize[1]:
            raise ValueError('expected square blocks in BSR matrix A')
    else:
        bsr_flag = False
    
    ##
    # Convert A to csc and Atilde to csr
    if isspmatrix_csr(A):
        Atilde = A.copy()
        A = A.tocsc()
    else:
        A = A.tocsc()
        Atilde = A.copy()
        Atilde = Atilde.tocsr()

    ##
    # Calculate the weighted-Jacobi parameter
    D = A.diagonal()
    Dinv = 1.0/D
    Dinv[D == 0] = 0.0
    Dinv = csc_matrix( (Dinv, (arange(A.shape[0]),arange(A.shape[1]))), shape=A.shape)
    DinvA = Dinv*A
    omega = 1.0/approximate_spectral_radius(DinvA, maxiter=20)
    del DinvA

    ## 
    # Approximate A-inverse with k steps of w-Jacobi and a zero initial guess
    S = csc_matrix( (array([]),(array([]),array([]))), shape=A.shape )
    I = speye(A.shape[0], A.shape[1], format='csc')
    for i in range(k+1):
        S = S + omega*(Dinv*(I - A*S))
    
    ##
    # Calculate the strength entries in S column-wise, but only strength
    # values at the sparsity pattern of A
    for i in range(Atilde.shape[0]):
        v = mat(S[:,i].todense())
        Av = mat(A*v)
        denom = sqrt(conjugate(v).T*Av)
        ##
        # replace entries in row i with strength values
        for j in range(Atilde.indptr[i], Atilde.indptr[i+1]):
            col = Atilde.indices[j]
            vj = v[col].copy()
            v[col] = 0.0
            #         =  (||v_j||_A - ||v||_A) / ||v||_A
            val = sqrt(conjugate(v).T*A*v)/denom - 1.0
            
            # Negative values generally imply a weak connection
            if val > -0.01:
                Atilde.data[j] = abs(val)
            else:
                Atilde.data[j] = 0.0

            v[col] = vj
 
    ##
    # Apply drop tolerance
    Atilde = classical_strength_of_connection(Atilde, theta=theta)
    Atilde.eliminate_zeros()

    ##
    # Put ones on the diagonal
    Atilde = Atilde + I.tocsr()
    Atilde.sort_indices()

    ##
    # Amalgamate Atilde for the BSR case, using ones for all strong connections
    if bsr_flag:
        Atilde = Atilde.tobsr(blocksize=(numPDEs, numPDEs))
        nblocks = Atilde.indices.shape[0]
        Atilde = csr_matrix( (ones((nblocks,)), Atilde.indices, Atilde.indptr), shape=(Atilde.shape[0]/numPDEs, Atilde.shape[1]/numPDEs) )
    
    return Atilde



def ode_strength_of_connection(A, B, epsilon=4.0, k=2, proj_type="l2"):
    """Construct an AMG strength of connection matrix using an ODE-based measure

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
    proj_type : {'l2','D_A'}
        Define norm for constrained min prob, i.e. define projection
   
    Returns
    -------
    Atilde : {csr_matrix}
        Sparse matrix of strength values

    References
    ----------
    .. [1] Olson, L. N., Schroder, J., Tuminaro, R. S., 
       "A New Perspective on Strength Measures in Algebraic Multigrid", 
       submitted, June, 2008.

    Examples
    --------
    >>> from numpy import array
    >>> from pyamg.gallery import stencil_grid
    >>> from pyamg.strength import ode_strength_of_connection
    >>> n=3
    >>> stencil = array([[-1.0,-1.0,-1.0],
    >>>                  [-1.0, 8.0,-1.0],
    >>>                  [-1.0,-1.0,-1.0]])
    >>> A = stencil_grid(stencil, (n,n), format='csr')
    >>> S = ode_strength_of_connection(A, ones((A.shape[0],1)))
    """
    # many imports for ode_strength_of_connection, so moved the imports local
    from numpy import array, zeros, mat, ravel, diff, mod, repeat,\
         inf, asarray, log2, finfo
    from scipy.sparse import bsr_matrix, spdiags, eye
    from scipy.sparse import eye as speye
    from pyamg.util.utils  import scale_rows

    #====================================================================
    #Check inputs
    if epsilon < 1.0:
        raise ValueError("expected epsilon > 1.0")
    if k <= 0:
        raise ValueError("number of time steps must be > 0")
    if proj_type not in ['l2', 'D_A']:
        raise VaueError("proj_type must be 'l2' or 'D_A'")
    if (not isspmatrix_csr(A)) and (not isspmatrix_bsr(A)):
        raise TypeError("expected csr_matrix or bsr_matrix") 
    
    #====================================================================
    # Format A and B correctly.
    #B must be in mat format, this isn't a deep copy
    Bmat = mat(B)

    # Amat must be in CSR format, be devoid of 0's and have sorted indices
    # Number of PDEs per point is defined implicitly by block size
    if not isspmatrix_csr(A):
       csrflag = False
       numPDEs = A.blocksize[0]
       A = A.tocsr()
    else:
        csrflag = True
        numPDEs = 1

    A.eliminate_zeros()
    A.sort_indices()

    #====================================================================
    # Handle preliminaries for the algorithm
    
    dimen = A.shape[1]
    NullDim = Bmat.shape[1]
        
    #Get spectral radius of Dinv*A, this will be used to scale the time step size for the ODE 
    D = A.diagonal();
    Dinv = 1.0 / D
    Dinv[D == 0] = 1.0
    Dinv_A  = scale_rows(A, Dinv, copy=True)
    rho_DinvA = approximate_spectral_radius(Dinv_A)
    
    #Calculate D_A for later use in the minimization problem
    if proj_type == "D_A":
        D_A = spdiags( [D], [0], dimen, dimen, format = 'csr')
    else:
        D_A = speye(dimen, dimen, format="csr")
    #====================================================================
    
    
    #====================================================================
    # Calculate (I - delta_t Dinv A)^k  
    #      In order to later access columns, we calculate the transpose in 
    #      CSR format so that columns will be accessed efficiently
    
    # Calculate the number of time steps that can be done by squaring, and 
    # the number of time steps that must be done incrementally
    nsquare = int(log2(k))
    ninc = k - 2**nsquare

    # Calculate one time step
    I = speye(dimen, dimen, format="csr")
    Atilde = (I - (1.0/rho_DinvA)*Dinv_A)
    Atilde = Atilde.T.tocsr()

    #Construct a sparsity mask for Atilde that will restrict Atilde^T to the 
    # nonzero pattern of A, with the added constraint that row i of Atilde^T 
    # retains only the nonzeros that are also in the same PDE as i. 
    mask = A.copy()
    
    # Restrict to same PDE
    if numPDEs > 1:
        row_length = diff(mask.indptr)
        my_pde = mod(range(dimen), numPDEs)
        my_pde = repeat(my_pde, row_length)
        mask.data[ mod(mask.indices, numPDEs) != my_pde ] = 0.0
        del row_length, my_pde
        mask.eliminate_zeros()

    # If total the number of time steps is a power of two, then there is  
    # a very efficient computational short-cut.  Otherwise, we support  
    # other numbers of time steps, through an inefficient algorithm.
    if False:
        # Use Chebyshev polynomial of order k
        # Poly coefficients, ignore the last (constant) coefficient
        from pyamg.relaxation.chebyshev import chebyshev_polynomial_coefficients
        bound = approximate_spectral_radius(A)
        coefficents = -chebyshev_polynomial_coefficients(bound/30.0, 1.1*bound, k)[:-1]

        # Generate Poly, coeff lists the coefficients in descending order
        Atilde = coefficents[0]*I
        for c in coefficents[1:]:
            Atilde = c*I + A*Atilde 
        
        #Apply mask to Atilde, zeros in mask have already been eliminated at start of routine.
        mask.data[:] = 1.0
        Atilde = Atilde.multiply(mask)
        Atilde.eliminate_zeros()
        Atilde.sort_indices()
        del mask

    elif ninc > 0: 
        print "The most efficient time stepping for the ODE Strength Method"\
              " is done in powers of two.\nYou have chosen " + str(k) + " time steps."
    
        # Calculate (Atilde^nsquare)^T = (Atilde^T)^nsquare
        for i in range(nsquare):
            Atilde = Atilde*Atilde
        
        JacobiStep = (I - (1.0/rho_DinvA)*Dinv_A).T.tocsr()
        for i in range(ninc):    
            Atilde = Atilde*JacobiStep
        del JacobiStep

        #Apply mask to Atilde, zeros in mask have already been eliminated at start of routine.
        mask.data[:] = 1.0
        Atilde = Atilde.multiply(mask)
        Atilde.eliminate_zeros()
        Atilde.sort_indices()

        ##Check matrix Atilde^k vs. above    
        #Atilde2 = (((I - (1.0/rho_DinvA)*Dinv_A).T)**k).multiply(mask)
        #differ = ravel((Atilde2 - Atilde).todense())
        #print "Sum(Abs(differ)) is " + str(sum(abs(differ)))       
        del mask

    elif nsquare == 0:
        if numPDEs > 1:
            #Apply mask to Atilde, zeros in mask have already been eliminated at start of routine.
            mask.data[:] = 1.0
            Atilde = Atilde.multiply(mask)
            Atilde.eliminate_zeros()
            Atilde.sort_indices()

        del mask

    else:
        # Use computational short-cut for case (ninc == 0) and (nsquare > 0)  
        # Calculate Atilde^k only at the sparsity pattern of mask.
        for i in range(nsquare-1):
            Atilde = Atilde*Atilde

        # Call incomplete mat-mat mult
        AtildeCSC = Atilde.tocsc() 
        AtildeCSC.sort_indices()
        mask.sort_indices()
        Atilde.sort_indices()
        multigridtools.incomplete_matmat(Atilde.indptr,    Atilde.indices,    Atilde.data, 
                                         AtildeCSC.indptr, AtildeCSC.indices, AtildeCSC.data,
                                         mask.indptr,      mask.indices,      mask.data,      
                                         dimen)
        
        del AtildeCSC, Atilde
        Atilde = mask            

    del Dinv, Dinv_A
           
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
        # Strength(i,j) = | 1 - (z(i)/b(j))/(z(j)/b(i)) |
        # These ratios can be calculated by diagonal row and column scalings
        
        #Create necessary Diagonal matrices
        DAtilde = Atilde.diagonal();
        DAtildeDivB = array(DAtilde)/array(Bmat).reshape(DAtilde.shape)
        DAtildeDivB[ravel(Bmat) == 0] = 1.0
        DAtildeDivB[DAtilde == 0] = 0.0
        DAtildeDivB = spdiags( [DAtildeDivB], [0], dimen, dimen, format = 'csr')
        DiagB = spdiags( [array(Bmat).flatten()], [0], dimen, dimen, format = 'csr')

        # Calculate best approximation in span(B)
        data = Atilde.data.copy()
        Atilde.data[:] = 1.0

        Atilde = DAtildeDivB*Atilde
        Atilde = Atilde*DiagB

        #Find negative ratios
        neg_ratios =  (sign(real(Atilde.data)) != sign(real(data))) + (sign(imag(Atilde.data)) != sign(imag(data)))

        #Calculate Approximation ratio
        Atilde.data = Atilde.data/data
        del data

        #Calculate Approximation error
        Atilde.data = abs( 1.0 - Atilde.data)
        
        #Drop negative ratios
        Atilde.data[neg_ratios] = 0.0
        Atilde.eliminate_zeros()
        
        #Set near perfect connections to 1e-4
        Atilde.data[Atilde.data < sqrt(finfo(float).eps)] = 1e-4

    else:
        # For use in computing local B_i^H*B, precompute the element-wise multiply of 
        #   each column of B with each other column.  We also scale by 2.0 
        #   to account for BDB's eventual use in a constrained minimization problem
        BDBCols = sum(range(NullDim+1))
        BDB = zeros((dimen,BDBCols))
        counter = 0
        for i in range(NullDim):
            for j in range(i,NullDim):
                BDB[:,counter] = 2.0*conjugate(ravel(asarray(B[:,i])))*ravel(asarray(D_A*B[:,j]))
                counter = counter + 1        
        
        # Use constrained min problem to define strength
        multigridtools.ode_strength_helper(Atilde.data,         Atilde.indptr,     Atilde.indices, 
                                           Atilde.shape[0],     ravel(asarray(B)), ravel(asarray((D_A*conjugate(B)).T)), 
                                           ravel(asarray(BDB)), BDBCols,           NullDim)
        
        Atilde.eliminate_zeros()
    #===================================================================
    
    #Apply drop tolerance
    if epsilon != inf:
        # All of the strength values are real by this point, so ditch the complex part
        Atilde.data = array(Atilde.data, dtype=float)
        multigridtools.apply_distance_filter(dimen, epsilon, Atilde.indptr, Atilde.indices, Atilde.data)
        Atilde.eliminate_zeros()
    
    # Set diagonal to 1.0, as each point is strongly connected to itself.
    I = speye(dimen, dimen, format="csr")
    I.data -= Atilde.diagonal()
    Atilde = Atilde + I

    # If converted BSR to CSR, convert back and return amalgamated matrix, 
    #   i.e. the sparsity structure of the blocks of Atilde
    if not csrflag:
        Atilde = Atilde.tobsr(blocksize=(numPDEs, numPDEs))
        
        n_blocks = Atilde.indices.shape[0]
        blocksize = Atilde.blocksize[0]*Atilde.blocksize[1]
        CSRdata = zeros((n_blocks,))
        multigridtools.min_blocks(n_blocks, blocksize, ravel(asarray(Atilde.data)), CSRdata)
        #Atilde = csr_matrix((data, row, col), shape=(*,*))
        Atilde = csr_matrix((CSRdata, Atilde.indices, Atilde.indptr), shape=(Atilde.shape[0]/numPDEs, Atilde.shape[1]/numPDEs) )
    
    return Atilde

