from numpy import empty, empty_like

from scipy.sparse import csr_matrix, isspmatrix_csr

from multilevel import multilevel_solver
import multigridtools

__all__ = ['ruge_stuben_solver','rs_strong_connections','rs_prolongator',
        'rs_cf_splitting', 'rs_direct_interpolation']


def ruge_stuben_solver(A, max_levels=10, max_coarse=500, cf='RS'):
    """
    Create a multilevel solver using Ruge-Stuben coarsening (Classical AMG)

    References:
        Trottenberg, U., C. W. Oosterlee, and Anton Schuller.
        "Multigrid"
        San Diego: Academic Press, 2001.
        Appendix A

    """
    As = [A]
    Ps = []
    Rs = []

    while len(As) < max_levels  and A.shape[0] > max_coarse:
        P = rs_prolongator(A, cf=cf)
        R = P.T.tocsr()

        A = R * A * P     #galerkin operator

        As.append(A)
        Ps.append(P)
        Rs.append(R)

    return multilevel_solver(As,Ps,Rs=Rs)



def rs_strong_connections(A,theta):
    """Return a strength of connection matrix using the method of Ruge and Stuben

        An off-diagonal entry A[i.j] is a strong connection iff

                -A[i,j] >= theta * max( -A[i,k] )   where k != i
    """
    if not isspmatrix_csr(A): raise TypeError('expected csr_matrix')

    Sp = empty_like(A.indptr)
    Sj = empty_like(A.indices)
    Sx = empty_like(A.data)

    multigridtools.rs_strong_connections( A.shape[0], theta, 
            A.indptr, A.indices, A.data, Sp,Sj,Sx)

    return csr_matrix((Sx,Sj,Sp),shape=A.shape)

def rs_cf_splitting(S):
    """Coarse-Fine splitting for strength of connection matrix S
    using the standard (serial) Ruge-Stuben algorithm

    """

    if not isspmatrix_csr(S): raise TypeError('expected csr_matrix')

    T = S.T.tocsr()  #transpose S for efficient column access

    splitting = empty( S.shape[0], dtype='intc' )

    multigridtools.rs_cf_splitting(S.shape[0],
            S.indptr, S.indices,  
            T.indptr, T.indices, 
            splitting)

    return splitting


def rs_direct_interpolation(A,S,splitting):
    if not isspmatrix_csr(S): raise TypeError('expected csr_matrix')

    Pp = empty_like( A.indptr )

    multigridtools.rs_direct_interpolation_pass1( A.shape[0],
            S.indptr, S.indices, splitting,  Pp)

    nnz = Pp[-1]
    Pj = empty( nnz, dtype=Pp.dtype )
    Px = empty( nnz, dtype=A.dtype )

    multigridtools.rs_direct_interpolation_pass2( A.shape[0],
            A.indptr, A.indices, A.data,
            S.indptr, S.indices, S.data,
            splitting,
            Pp,       Pj,        Px)

    return csr_matrix( (Px,Pj,Pp) )



def rs_prolongator(A, theta=0.25, cf='RS'):
    if not isspmatrix_csr(A): raise TypeError('expected csr_matrix')

    S = rs_strong_connections(A,theta)

    if cf == 'RS':
        splitting = rs_cf_splitting(S)
    elif cf in ['PMISc']:
        import split
        splitting = getattr(split,cf)(S)
    else:
        raise ValueError('unknown C/F splitting method (%s)' % cf)

    return rs_direct_interpolation(A,S,splitting)

    #T = S.T.tocsr()  #transpose S for efficient column access

    #Ip,Ij,Ix = multigridtools.rs_interpolation(A.shape[0],\
    #                                           A.indptr,A.indices,A.data,\
    #                                           S.indptr,S.indices,S.data,\
    #                                           T.indptr,T.indices,T.data)

    #return csr_matrix((Ix,Ij,Ip))
