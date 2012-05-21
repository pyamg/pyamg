"""
Test the convergence for a simple 100x100 Grid, Gauge Laplacian Matrix

For this problem, the matrix A is complex, but this isn't problematic,
because complex arithmetic is natively supported.  There is _no_ 
implicit conversion to an equivalent real system.

"""
import numpy
import scipy
from pyamg.gallery import gauge_laplacian 
from pyamg import smoothed_aggregation_solver, rootnode_solver
from convergence_tools import print_cycle_history

if __name__ == '__main__':
    n = 100
    
    print "Test convergence for a simple 100x100 Grid, Gauge Laplacian"
    choice = input('\n Input Choice:\n' + \
               '1:  Run smoothed_aggregation_solver\n' + \
               '2:  Run rootnode_solver\n' )

    numpy.random.seed(625)
    A = gauge_laplacian(n, beta=0.001) 
    x = scipy.rand(A.shape[0]) + 1.0j*scipy.rand(A.shape[0])
    b = scipy.rand(A.shape[0]) + 1.0j*scipy.rand(A.shape[0])
    
    if choice == 1:
        sa = smoothed_aggregation_solver(A, smooth='energy')
    elif choice == 2:
        sa = rootnode_solver(A, smooth='energy')
    else:
        raise ValueError("Enter a choice of 1 or 2")

    resvec = []
    x = sa.solve(b, x0=x, maxiter=20, tol=1e-14, residuals=resvec)

    print_cycle_history(resvec, sa, verbose=True, plotting=True)


