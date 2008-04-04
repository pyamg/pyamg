"""Tentative prolongator"""

__docformat__ = "restructuredtext en"

from numpy import zeros, sqrt, asarray, empty, diff
from scipy.sparse import isspmatrix_csr, bsr_matrix
from pyamg.utils import scale_columns

__all__ = ['fit_candidates']


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


