
"""
run with
mprof run testmemory
mprof plot
"""

from memory_profiler import profile

@profile
def prof():
    import numpy as np
    import pyamg
    np.random.seed(2017)

    stencil = [ [-1,-1,-1],[-1,8,-1],[-1,-1,-1] ]
    A = pyamg.gallery.stencil_grid(stencil, (100,100), dtype=float, format='csr')
    near_null_space = np.ones(A.shape[0])
    ml = pyamg.smoothed_aggregation_solver(A, near_null_space[:, np.newaxis])

    for i in range(10000):
        print(i)
        rhs = np.random.randn(A.shape[0])
        x0 = np.random.randn(A.shape[0])
        solution = ml.solve(b=rhs.flatten(), x0=x0.flatten(), accel="bicgstab")

if __name__ == '__main__':
    prof()
