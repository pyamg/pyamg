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

    stencil = [ [-1,-1,-1],[-1,8,-1],[-1,-1,-1] ]
    n = 25
    A = pyamg.gallery.stencil_grid(stencil, (n,n), dtype=float, format='csr')
    ml = pyamg.smoothed_aggregation_solver(A, max_levels=3)

    b = np.random.randn(A.shape[0])
    x = np.zeros(A.shape[0])

    for i in range(2000000):
        print(i)
        #solution = ml.solve(b)
        pyamg.relaxation.relaxation.gauss_seidel(A, x, b)

if __name__ == '__main__':
    prof()
