import pyamg
import numpy as np
from timeit import default_timer as timer

n = 1000
nx = n
ny = n

stencil = pyamg.gallery.diffusion_stencil_2d(
    type='FE', epsilon=0.001, theta=np.pi / 3)

A = pyamg.gallery.stencil_grid(stencil, (nx, ny), format='csr')

u = np.random.rand(A.shape[0])

t0 = timer()
v = A * u
t1 = timer()
print('time {}'.format(t1-t0))

A2 = pyamg.util.sparse.csr(A)

t0 = timer()
v = A * u
t1 = timer()
print('time {}'.format(t1-t0))
