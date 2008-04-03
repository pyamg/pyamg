"""Strength of Connection functions"""

__docformat__ = "restructuredtext en"

from numpy import ones, empty_like

from scipy.sparse import csr_matrix, isspmatrix_csr, isspmatrix_bsr
import multigridtools

__all__ = ['classical_strength_of_connection', 'symmetric_strength_of_connection']


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


