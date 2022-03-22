import pyamg
import numpy as np
from timeit import default_timer as timer

n = 300

t0 = timer()
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
