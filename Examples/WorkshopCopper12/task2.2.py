from pyamg import gallery, smoothed_aggregation_solver
from numpy import ones
from pylab import *

A = gallery.poisson( (100,100), format='csr')
ml = smoothed_aggregation_solver(A, smooth='simple')
b = ones((A.shape[0],1)); 
res=[]
x = ml.solve(b, tol=1e-8, residuals=res)
semilogy(res[1:])
xlabel('iteration')
ylabel('residual norm')
show()
