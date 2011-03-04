#import cg
import _cg as cg
import precondition
import gallery
import numpy
from scipy.sparse import triu, tril, spdiags, eye
from scipy.sparse.linalg import LinearOperator
import time

def norm(x):
    x = x.ravel()
    return numpy.sqrt( numpy.inner(x.conj(),x).real )

n    = 25
A    = gallery.poisson((n,n)).tocsr()
N = A.shape[0]
xx   = numpy.arange(1,N+1)
b    = A*xx
x0   = numpy.zeros((N,))

print "==================="
print " Running PCG with %d x %d Poisson\n"%(N,N)
print " Accel  Stats"
print "-----------------------------"
precs = (None,'J', 'SGS','SSOR')
omega = 2.0/3
maxiter = 500
for p in precs:
    M = precondition.basic(A,p,omega=omega)

    res=[]
    t1 = time.time()
    (x,flag) = cg.cg(A, b, x0=x0, tol=1e-8, maxiter=maxiter, M = M, residuals=res)
    t2 = time.time()
    cgtime = t2 - t1
    res = numpy.array(res)

    print "%5s \titers   = %d"%(p,len(res))
    print "      \ttime      = %1.4g ms"%((cgtime)*1000)
    print "      \ttime/iter = %1.4g ms\n"%((cgtime)*1000/len(res))

    try:
        import pylab
        pylab.semilogy(res,'-')
        pylab.axis([0,maxiter,0,10*norm(b)])
        if p==precs[-1]:
            pylab.legend([str(precs[i]) for i in range(len(precs)) ])
            pylab.ylabel('residual norm')
            pylab.xlabel('iteration')
    except:
        continue

t1 = time.time()
z = M*b
t2 = time.time()
print "time for one LU solve  = %1.4g ms\n"%((t2-t1)*1000)

try:
    import pylab
    pylab.show()
except:
    print "Pylab not installed...skipping plotting\n"


