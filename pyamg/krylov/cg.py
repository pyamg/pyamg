from numpy import asmatrix, asarray, zeros, dot
from scipy.linalg import norm
from scipy.linalg import solve as direct_solve
from numpy.random import random
from types import FunctionType

__all__ = ['cg']

def checkinput(A,b,x0,M,Mopts):
    try:
        A = asarray(A)
        b = asarray(b)
        x0 = asarray(x0)
    except:
        raise ValueError, 'expect arrays for A, b, and x0'

    # check sizes of A, b, x0

    if A.shape[0] != A.shape[1]:
        raise ValueError, 'CG expects square matrix'

    if A.shape[0] != len(b) or A.shape[0] != len(x0):
        raise ValueError, 'CG expects Ax=b with consistent shape'

    b = b.reshape((len(b),1))
    x0 = x0.reshape((len(b),1))

    # check size and functionality of M
    if type(M) == FunctionType:
        Mfunc = M
        try:
            x = Mfunc(x0,**Mopts)
        except:
            raise RuntimeError, 'problem with preconditioner function'

        if len(x) != len(b):
            raise ValueError, 'CG expects Mx with consistent shape'
    else:
        try:
            M = asmatrix(M)
        except:
            raise ValueError, 'expect arrays for A and b'

        if A.shape[0] != A.shape[1] or A.shape[0] != len(b):
            raise ValueError, 'CG expects square matrix M of size of A'

def cg(A, b, x0=None, tol=1e-6, maxit=10, M=None, Mopts=None):
    """
    PCG from Saad
    """

    # check input
    if x0==None:
        x0 = random((len(b),1))
    x = zeros((len(b),1))
    checkinput(A,b,x0,x,M)

    A = asarray(A)
    b = asarray(b)
    x0 = asarray(x0)
   
    # setup
    doneiterating = False
    iter = 0

    r = b - A * b
    normr0 = norm(r)

    if normr0 < tol:
        doneiterating = True

    if type(M) == FunctionType:
        Mfunc = M
        z = Mfunc(r,**Mopts)
    else:
        if M==None:
            z = r
        else:
            z = direct_solve(M,r)

    p = z.copy()
    Ap = dot(A,p)

    while not doneiterating:
        alpha = dot(r,z)/dot(Ap,p)
        x = x + alpha * p
        rnew = r - alpha * Ap

        if type(M) == FunctionType:
            znew = Mfunc(rnew,**Mopts)
        else:
            if M==None:
                znew = rnew
            else:
                znew = direct_solve(M,rnew)

        beta = dot(rnew,znew)/dot(r,z)
        p = znew + beta * p

        z = znew
        r = rnew
        Ap = dot(A,p)
        iter += 1

        normr = norm(r)
        if normr/normr0 < tol:
            doneiterating = True

        if iter>(maxit-1):
            doneiterating = True

        print iter
        print normr/normr0
    return x

if __name__ == '__main__':
    from numpy import diag
    A = random((4,4))
    A = A*A.transpose() + diag([10,10,10,10])
    b = random((4,1))
    x0 = random((4,1))

    from pyamg.gallery import stencil_grid
    stencil_grid([[0,-1,0],[-1,4,-1],[0,-1,0]],((100,100)),dtype=float,format='csr')
    b = random((A.shape[0],1))

    def myM(x0,str):
        print str
        return 2*x0

    x = cg(A,b,maxit=100)
    print norm(b - dot(A,x))

    #cg(A,b,x0=None,M=myM,Mopts={'str':'here'})
