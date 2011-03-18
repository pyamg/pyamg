from pyamg import gallery, smoothed_aggregation_solver
from numpy import ones
from pylab import *

def new_relax(A,x,b):
    x[:] +=  0.125*(b - A*x)

A = gallery.poisson( (100,100), format='csr')
b = ones( (A.shape[0],1))
res = []
ml = smoothed_aggregation_solver(A)
ml.levels[0].presmoother = new_relax
ml.levels[0].postsmoother = new_relax
x = ml.solve(b, tol=1e-8, residuals=res)
semilogy(res[1:])
xlabel('iteration')
ylabel('residual norm')
show()
