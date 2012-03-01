# task1.3
from pyamg.gallery.diffusion import diffusion_stencil_2d
from pyamg.gallery import stencil_grid
sten = diffusion_stencil_2d(type='FD', \
       epsilon=0.001, theta=3.1416/3.0)
A = stencil_grid(sten, (100,100), format='csr')

# task1.6
from pyamg import *
from numpy import ones
ml = smoothed_aggregation_solver(A, \
     strength='evolution',          \
     smooth=('energy', {'degree':4}) )
b = ones((A.shape[0],1))
res = []
x = ml.solve(b, tol=1e-8, residuals=res)
from pylab import *
semilogy(res[1:])
xlabel('iteration')
ylabel('residual norm')
title('Residual History')
show()
