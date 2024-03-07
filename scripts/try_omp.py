import pyamg
import numpy as np
from timeit import default_timer as timer

n = 1000

t0 = timer()
A = pyamg.gallery.poisson((n,n), format='csr')
ml = pyamg.smoothed_aggregation_solver(A, max_coarse=10)
b = np.random.rand(A.shape[0])
u = np.random.rand(A.shape[0])
t1 = timer()
print(f'problem size={A.shape}')
print(f'setup time {t1-t0}')

t0 = timer()
res = []
x = ml.solve(b, x0=u, residuals=res)
t1 = timer()
print(f'reg time {t1-t0}')
print(res, '\n')

t0 = timer()
res = []
x = ml.solve(b, x0=u, openmp=True, residuals=res)
t1 = timer()
print(f'omp time {t1-t0}')
print(res, '\n')
