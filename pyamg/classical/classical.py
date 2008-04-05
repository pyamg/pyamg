"""Classical AMG (Ruge-Stuben AMG)"""

__docformat__ = "restructuredtext en"

from numpy import empty, empty_like

from scipy.sparse import csr_matrix, isspmatrix_csr

from pyamg.multilevel import multilevel_solver
from pyamg.strength import classical_strength_of_connection
from pyamg import multigridtools

__all__ = ['ruge_stuben_solver','rs_direct_interpolation']


def ruge_stuben_solver(A, theta=0.25, CF='RS', 
        max_levels=10, max_coarse=500, **kwargs):
    """Create a multilevel solver using Classical AMG (Ruge-Stuben AMG)

    Parameters
    ----------
    A : {csr_matrix, bsr_matrix}
        Square matrix in CSR or BSR format
    theta : {float} : default 0.25
        Strength of connection parameter
    CF : {string} : default 'RS'
        Method used for coarse grid selection (C/F splitting)
        Supported methods are RS, PMIS, PMISc, CLJP, and CLJPc
    max_levels: {integer} : default 10
        Maximum number of levels to be used in the multilevel solver.
    max_coarse: {integer} : default 500
        Maximum number of variables permitted on the coarse grid.

    References
    ----------
        Trottenberg, U., C. W. Oosterlee, and Anton Schuller.
        "Multigrid"
        San Diego: Academic Press, 2001.
        Appendix A

    """

    class rs_level:
        pass

    levels = []
    
    while len(levels) < max_levels  and A.shape[0] > max_coarse:
        S,splitting,P = prolongator(A, theta=theta, CF=CF)

        R = P.T.tocsr()

        levels.append( rs_level() )
        levels[-1].A = A
        levels[-1].S = S                  # strength of connection matrix
        levels[-1].P = P                  # prolongation operator
        levels[-1].R = R                  # restriction operator
        levels[-1].spliting = splitting

        A = R * A * P                     #galerkin operator

    levels.append( rs_level() )
    levels[-1].A = A

    return multilevel_solver(levels, **kwargs)

#TODO rename and move
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


def prolongator(A, theta, CF):
    if not isspmatrix_csr(A): raise TypeError('expected csr_matrix')

    S = classical_strength_of_connection(A,theta)

    if CF in [ 'RS', 'PMIS', 'PMISc', 'CLJP', 'CLJPc']:
        import split
        splitting = getattr(split, CF)(S)
    else:
        raise ValueError('unknown C/F splitting method (%s)' % CF)

    P = rs_direct_interpolation(A,S,splitting)

    return S,splitting,P

