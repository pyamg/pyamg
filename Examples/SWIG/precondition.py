from scipy.sparse import csr_matrix, isspmatrix_csr, speye
from scipy.sparse import triu, tril, spdiags, eye
from scipy.sparse.linalg import LinearOperator
from numpy import zeros, dot
import splinalg

__docformat__ = "restructuredtext en"

__all__ = ['lusolve']

def lusolve(L,U,r):
    """
       M z = r
     L U z = r
         z = U^-1 L^-1 r
    1. solve L w = r
    2. solve U z = w

    c++ implementation (60x faster)
    """

    if not isspmatrix_csr(L) or not isspmatrix_csr(U):
        print "Must be CSR"

    n = L.shape[0]
    w = zeros((n,))
    z = zeros((n,))

    splinalg.forwardsolve(L.indptr,L.indices,L.data,w,r,n)
    splinalg.backsolve(U.indptr,U.indices,U.data,z,w,n)
    return z

def lusolve_reference(L,U,r):
    """
       M z = r
     L U z = r
         z = U^-1 L^-1 r
    1. solve L w = r
    2. solve U z = w

    reference implementation
    """

    if not isspmatrix_csr(L) or not isspmatrix_csr(U):
        print "Must be CSR"

    n = L.shape[0]
    w = zeros((n,))
    z = zeros((n,))
    
    # Forward solve
    i=0
    while i<n:
        a = L.indptr[i]
        b = L.indptr[i+1]-1 # everything but the diag which is last
        d = L.data[b]
        w[i] = (r[i] - dot(L.data[a:b],w[L.indices[a:b]]))/d
        i+=1

    # Backward solve
    i=n-1
    while i>-1:
        a = U.indptr[i]+1   # everything but the diag which is first
        b = U.indptr[i+1]
        d = U.data[a-1]
        z[i] = (w[i] - dot(U.data[a:b],z[U.indices[a:b]]))/d
        i-=1
        
    return z


def preconditioner_matvec(L,U):
    def matvec(x):
        return lusolve_reference(L,U,x)
    return LinearOperator(shape=L.shape,matvec=matvec)

def basic(A,pname=None,omega=1.0):
    """ Jacobi, Gauss-Seidel, SOR and symmetric version

    J, GS, SGS, SOR, SSOR
    """
    n = A.shape[0]
    
    if pname is None:
        I    = eye(n,n).tocsr()
        L = I
        U = I
    elif pname=='J':
        # weigthed Jacobi
        I = eye(n,n).tocsr()
        D = spdiags(A.diagonal(),0,n,n).tocsr()
        #
        L = omega * D
        U = I
    elif pname=='GS':
        # Gauss Seidel
        I = eye(n,n).tocsr()
        D = spdiags(A.diagonal(),0,n,n).tocsr()
        E = tril(A,-1).tocsr()
        #
        L = D + E 
        U = I
    elif pname=='SGS':
        # Symmetric Gauss-Seidel
        D = spdiags(A.diagonal(),0,n,n).tocsr()
        Dinv =  spdiags(1/A.diagonal(),0,n,n).tocsr()
        E = tril(A,-1).tocsr()
        F = triu(A,1).tocsr()
        #
        L = (D+E)*Dinv
        U = D+F
    elif pname=='SOR':
        # Successive Overrelaxation
        I = eye(n,n).tocsr()
        D = spdiags(A.diagonal(),0,n,n).tocsr()
        E = tril(A,-1).tocsr()
        #
        L = D + omega * E 
        U = I
    elif pname=='SSOR':
        # Symmetric Successive Overrelaxation
        D = spdiags(A.diagonal(),0,n,n).tocsr()
        Dinv =  spdiags(1/A.diagonal(),0,n,n).tocsr()
        E = tril(A,-1).tocsr()
        F = triu(A,1).tocsr()
        #
        L = (D+omega*E)*Dinv
        U = D+omega*F
    else:
        print ">>>>>>>>>>>>>    Problem with the preconditioner name..."
        I    = eye(n,n).tocsr()
        L = I
        U = I
    
    L.sort_indices()
    U.sort_indices()
    return preconditioner_matvec(L,U)
