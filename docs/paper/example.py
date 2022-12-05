import pyamg
import numpy as np

np.random.seed(2022)
n = 10000
A = pyamg.gallery.poisson((n, n), format='csr')
ml = pyamg.smoothed_aggregation_solver(A, max_coarse=10)
print(ml)

x0 = np.random.rand(A.shape[0])
b = np.zeros(A.shape[0])
res = []
x = ml.solve(b, x0, tol=1e-10, residuals=res)
res = np.array(res)
print(res[1:]/res[:-1])

np.savetxt('example.res.txt', res, header=str(ml))
