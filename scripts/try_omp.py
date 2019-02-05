import pyamg
import numpy as np
from timeit import default_timer as timer

n = 100
nx = n
ny = n

t0 = timer()
#stencil = pyamg.gallery.diffusion_stencil_2d(
#    type='FE', epsilon=0.001, theta=np.pi / 3)
#A = pyamg.gallery.stencil_grid(stencil, (nx, ny), format='csr')
A = pyamg.gallery.poisson((n,n,n), format='csr')
ml = pyamg.smoothed_aggregation_solver(A, max_coarse=10)
b = np.random.rand(A.shape[0])
u = np.random.rand(A.shape[0])
t1 = timer()
print('setup time {}'.format(t1-t0))

t0 = timer()
x = ml.solve(b, x0=u)
t1 = timer()
print('reg time {}'.format(t1-t0))

t0 = timer()
x = ml.solve(b, x0=u, openmp=True)
t1 = timer()
print('omp time {}'.format(t1-t0))
