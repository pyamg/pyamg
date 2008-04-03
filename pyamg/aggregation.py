"""Support for aggregation-based AMG"""

__docformat__ = "restructuredtext en"

from numpy import array, arange, ones, zeros, sqrt, asarray, \
        empty, empty_like, diff

from scipy.sparse import csr_matrix, coo_matrix, \
        isspmatrix_csr, bsr_matrix, isspmatrix_bsr

#from sa_ode_strong_connections import sa_ode_strong_connections
#from sa_energy_min import sa_energy_min

import multigridtools
from multilevel import multilevel_solver
from strength import symmetric_strength_of_connection
from utils import diag_sparse, approximate_spectral_radius, \
                  symmetric_rescaling, scale_columns, scale_rows

__all__ = ['smoothed_aggregation_solver', 'sa_filtered_matrix',
        'standard_aggregation', 'sa_smoothed_prolongator', 'fit_candidates']



def sa_filtered_matrix(A,epsilon):
    """The filtered matrix is obtained from A by lumping all weak off-diagonal
    entries onto the diagonal.  Weak off-diagonals are determined by
    the standard strength of connection measure using the parameter epsilon.

    In the case epsilon = 0.0, (i.e. no weak connections) A is returned.
    """

    if epsilon == 0:
        return A

    if isspmatrix_csr(A): 
        #TODO rework this
        raise NotImplementedError,'blocks not handled yet'
        Sp,Sj,Sx = multigridtools.symmetric_strength_of_connection(A.shape[0],epsilon,A.indptr,A.indices,A.data)
        return csr_matrix((Sx,Sj,Sp),shape=A.shape)
    elif ispmatrix_bsr(A):
        raise NotImplementedError,'blocks not handled yet'
    else:
        return sa_filtered_matrix(csr_matrix(A),epsilon)
##            #TODO subtract weak blocks from diagonal blocks?
##            num_dofs   = A.shape[0]
##            num_blocks = blocks.max() + 1
##
##            if num_dofs != len(blocks):
##                raise ValueError,'improper block specification'
##
##            # for non-scalar problems, use pre-defined blocks in aggregation
##            # the strength of connection matrix is based on the 1-norms of the blocks
##
##            B  = csr_matrix((ones(num_dofs),blocks,arange(num_dofs + 1)),shape=(num_dofs,num_blocks))
##            Bt = B.T.tocsr()
##
##            #1-norms of blocks entries of A
##            Block_A = Bt * csr_matrix((abs(A.data),A.indices,A.indptr),shape=A.shape) * B
##
##            S = symmetric_strength_of_connection(Block_A,epsilon)
##            S.data[:] = 1
##
##            Mask = B * S * Bt
##
##            A_strong = A ** Mask
##            #A_weak   = A - A_strong
##            A_filtered = A_strong

    return A_filtered


def standard_aggregation(C):
    """Compute the sparsity pattern of the tentative prolongator

    Parameters
    ----------
    C : csr_matrix
        strength of connection matrix

    Returns
    -------
    T : csr_matrix
        stores the sparsity pattern of the tentative prolongator


    Example
    -------
    TODO e
    """

    if not isspmatrix_csr(C): 
        raise TypeError('expected csr_matrix') 

    if C.shape[0] != C.shape[1]:
        raise ValueError('expected square matrix')

    index_type = C.indptr.dtype
    num_rows   = C.shape[0]

    Tj = empty( num_rows, dtype=index_type ) #stores the aggregate #s
    
    fn = multigridtools.standard_aggregation

    num_aggregates = fn(num_rows,C.indptr,C.indices,Tj)

    if num_aggregates == 0:
        return csr_matrix( (num_rows,1), dtype='int8' ) # all zero matrix
    else:
        shape = (num_rows, num_aggregates)
        if Tj.min() == -1:
            # some nodes not aggregated
            mask = Tj != -1
            row  = arange( num_rows, dtype=index_type )[mask]
            col  = Tj[mask]
            data = ones(len(col), dtype='int8')
            return coo_matrix( (data,(row,col)), shape=shape).tocsr()
        else:
            # all nodes aggregated
            Tp = arange( num_rows+1, dtype=index_type)
            Tx = ones( len(Tj), dtype='int8')
            return csr_matrix( (Tx,Tj,Tp), shape=shape)



def fit_candidates(AggOp, B, tol=1e-10):
    """Fit near-nullspace candidates to form the tentative prolongator

    Parameters
    ----------
    AggOp : csr_matrix
        Describes the sparsity pattern of the tentative prolongator.
        Has dimension (#blocks, #aggregates)
    B : array
        The near-nullspace candidates stored in columnwise fashion.
        Has dimension (#blocks * blocksize, #candidates)
    tol : scalar
        Threshold for eliminating local basis functions.
        If after orthogonalization a local basis function Q[:,j] is small, 
        i.e. ||Q[:,j]|| < tol, then Q[:,j] is set to zero.

    Returns
    -------
    (Q,R) : (bsr_matrix,array)
        The tentative prolongator Q is a sparse block matrix with dimensions 
        (#blocks * blocksize, #aggregates * #candidates) formed by dense blocks
        of size (blocksize, #candidates).  The coarse level candidates are 
        stored in R which has dimensions (#blocksize * #aggregates, #blocksize).

    Notes
    -----
        Assuming that each row of AggOp contains exactly one non-zero entry,
        i.e. all unknowns belong to an aggregate, then Q and R statisfy the 
        relationship B = Q*R.  In other words, the near-nullspace candidates
        are represented exactly by the tentative prolongator.

        If AggOp contains rows with no non-zero entries, then the range of the
        tentative prolongator will not include those degrees of freedom. This
        situation is illustrated in the examples below.

    
    Examples
    --------

    TODO: 
        Show B=[1,1,1,1]^T
        Show B=[[1,1,1,1],[0,1,2,3]]^T
        Show B=[1,1,1,1]^T where AggOp has zero row

    """
    if not isspmatrix_csr(AggOp):
        raise TypeError('expected csr_matrix for argument AggOp')

    if B.dtype != 'float32':
        B = asarray(B,dtype='float64')

    if len(B.shape) != 2:
        raise ValueError('expected rank 2 array for argument B')

    if B.shape[0] % AggOp.shape[0] != 0:
        raise ValueError('dimensions of AggOp %s and B %s are incompatible' % (AggOp.shape, B.shape))
    

    K = B.shape[1] # number of near-nullspace candidates
    blocksize = B.shape[0] / AggOp.shape[0]

    N_fine,N_coarse = AggOp.shape

    R = zeros((N_coarse,K,K), dtype=B.dtype) #storage for coarse candidates

    candidate_matrices = []

    for i in range(K):
        c = B[:,i]
        c = c.reshape(-1,blocksize,1)[diff(AggOp.indptr) == 1]     # eliminate DOFs that aggregation misses

        X = bsr_matrix( (c, AggOp.indices, AggOp.indptr), \
                shape=(blocksize*N_fine, N_coarse) )

        col_thresholds = tol * bsr_matrix((X.data**2,X.indices,X.indptr),shape=X.shape).sum(axis=0).A.flatten() 

        #orthogonalize X against previous
        for j,A in enumerate(candidate_matrices):
            D_AtX = bsr_matrix((A.data*X.data,X.indices,X.indptr),shape=X.shape).sum(axis=0).A.flatten() #same as diagonal of A.T * X
            R[:,j,i] = D_AtX
            X.data -= scale_columns(A,D_AtX).data

        #normalize X
        col_norms = bsr_matrix((X.data**2,X.indices,X.indptr),shape=X.shape).sum(axis=0).A.flatten() #same as diagonal of X.T * X
        mask = col_norms <= col_thresholds   # set small basis functions to 0

        col_norms = sqrt(col_norms)
        col_norms[mask] = 0
        R[:,i,i] = col_norms
        col_norms = 1.0/col_norms
        col_norms[mask] = 0

        scale_columns(X,col_norms,copy=False)

        candidate_matrices.append(X)

    Q_indptr  = AggOp.indptr
    Q_indices = AggOp.indices
    Q_data = empty((AggOp.nnz,blocksize,K)) #if AggOp includes all nodes, then this is (N_fine * K)
    for i,X in enumerate(candidate_matrices):
        Q_data[:,:,i] = X.data.reshape(-1,blocksize)
    Q = bsr_matrix((Q_data,Q_indices,Q_indptr),shape=(blocksize*N_fine,K*N_coarse))

    R = R.reshape(-1,K)

    return Q,R

def sa_smoothed_prolongator(A,T,epsilon=0.0,omega=4.0/3.0):
    """For a given matrix A and tentative prolongator T return the
    smoothed prolongator P
    
        P = (I - omega/rho(S) S) * T

    where S is a Jacobi smoothing operator defined as follows:
        omega      - damping parameter
        rho(S)     - spectral radius of S (estimated)
        S          - inv(diag(A_filtered)) * A_filtered   (Jacobi smoother)
        A_filtered - sa_filtered_matrix(A,epsilon)

    """

    A_filtered = sa_filtered_matrix(A,epsilon) #use filtered matrix for anisotropic problems

    # TODO use scale_rows()
    D = A_filtered.diagonal()
    D_inv = 1.0 / D
    D_inv[D == 0] = 0

    D_inv_A = scale_rows(A, D_inv, copy=True)
    D_inv_A *= omega/approximate_spectral_radius(D_inv_A)

    # smooth tentative prolongator T
    P = T - (D_inv_A*T)

    return P


def sa_prolongator(A, B, strength='standard', aggregate='standard', smooth='standard'):

    def unpack_arg(v):
        if isinstance(v,tuple):
            return v[0],v[1]
        else:
            return v,{}

    # strength of connection
    fn, kwargs = unpack_arg(strength)
    if fn == 'standard':
        C = symmetric_strength_of_connection(A,**kwargs)
    elif fn == 'ode':
        C = sa_ode_strong_connections(A,B,**kwargs)
    else:
        raise ValueError('unrecognized strength of connection method: %s' % fn)

    # aggregation
    fn, kwargs = unpack_arg(aggregate)
    if fn == 'standard':
        AggOp = standard_aggregation(C,**kwargs)
    else:
        raise ValueError('unrecognized aggregation method' % fn )

    # tentative prolongator
    T,B = fit_candidates(AggOp,B)

    # tentative prolongator smoother
    fn, kwargs = unpack_arg(smooth)
    if fn == 'standard':
        P = sa_smoothed_prolongator(A,T,**kwargs)
    elif fn == 'energy_min':
        P = sa_energy_min(A,T,C,B,**kwargs)
    else:
        raise ValueError('unrecognized prolongation smoother method % ' % fn)
    
    return P,B






def smoothed_aggregation_solver(A, B=None, max_levels = 10, max_coarse = 500,
                                solver = multilevel_solver, **kwargs):
    """Create a multilevel solver using Smoothed Aggregation (SA)

    Parameters
    ----------

    A : {csr_matrix, bsr_matrix}
        Square matrix in CSR or BSR format
    B : {None, array_like}
        Near-nullspace candidates stored in the columns of an NxK array.
        The default value B=None is equivalent to B=ones((N,1))
    max_levels: {integer} : default 10
        Maximum number of levels to be used in the multilevel solver.
    max_coarse: {integer} : default 500
        Maximum number of variables permitted on the coarse grid.
    
    Optional Parameters
    -------------------
    strength : strength of connection method
        Possible values are:
            'standard' 
            'ode'
    
    aggregate : aggregation method
        Possible values are:
            'standard'
    
    smooth : prolongation smoother
        Possible values are:
            'standard'
            'energy_min'


    Unused Parameters
        epsilon: {float} : default 0.0
            Strength of connection parameter used in aggregation.
        omega: {float} : default 4.0/3.0
            Damping parameter used in prolongator smoothing (0 < omega < 2)
        symmetric: {boolean} : default True
            True if A is symmetric, False otherwise
        rescale: {boolean} : default True
            If True, symmetrically rescale A by the diagonal
            i.e. A -> D * A * D,  where D is diag(A)^-0.5
        aggregation: {None, list of csr_matrix} : optional
            List of csr_matrix objects that describe a user-defined
            multilevel aggregation of the variables.
            TODO ELABORATE

    Example
    -------
        TODO

    References
    ----------

        Petr Vanek and Jan Mandel and Marian Brezina
        "Algebraic Multigrid by Smoothed Aggregation for Second and Fourth Order Elliptic Problems",
        http://citeseer.ist.psu.edu/vanek96algebraic.html

    """

    A = A.asfptype()

    if not (isspmatrix_csr(A) or isspmatrix_bsr(A)):
        raise TypeError('argument A must have type csr_matrix or bsr_matrix')

    if A.shape[0] != A.shape[1]:
        raise ValueError('expected square matrix')

    if B is None:
        B = ones((A.shape[0],1),dtype=A.dtype) # use constant vector
    else:
        B = asarray(B,dtype=A.dtype)

    pre,post = None,None   #preprocess/postprocess

    #if rescale:
    #    D_sqrt,D_sqrt_inv,A = symmetric_rescaling(A)
    #    D_sqrt,D_sqrt_inv = diag_sparse(D_sqrt),diag_sparse(D_sqrt_inv)

    #    B = D_sqrt * B  #scale candidates
    #    def pre(x,b):
    #        return D_sqrt*x,D_sqrt_inv*b
    #    def post(x):
    #        return D_sqrt_inv*x

    As = [A]
    Ps = []
    Rs = []

    while len(As) < max_levels and A.shape[0] > max_coarse:
        P,B = sa_prolongator(A,B,**kwargs)

        R = P.T.asformat(P.format)

        A = R * A * P     #galerkin operator

        As.append(A)
        Rs.append(R)
        Ps.append(P)

    #Check for all 0 coarse level.  Delete if found.
    if(A.nnz == 0):
    	As.pop(); Rs.pop(); Ps.pop();

    return solver(As,Ps,Rs=Rs,preprocess=pre,postprocess=post)


