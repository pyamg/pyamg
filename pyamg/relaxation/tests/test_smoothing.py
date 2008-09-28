from pyamg.testing import *

from pyamg.gallery    import poisson
from pyamg import smoothed_aggregation_solver
from pyamg.utils import profile_solver
    
methods = ['gauss_seidel',
           'jacobi',
           'richardson',
           'sor',
           #'chebyshev',
           'kaczmarz_gauss_seidel',
           'kaczmarz_jacobi',
           'kaczmarz_richardson']

class TestSmoothing(TestCase):
    def test_solver_parameters(self):
        A = poisson((50,50), format='csr')

        for method in methods:
            #method = ('richardson', {'omega':4.0/3.0})
            ml = smoothed_aggregation_solver(A, presmoother=method, postsmoother=method, max_coarse=10)

            residuals = profile_solver(ml)
            print "method",method
            print "residuals",residuals
            print "convergence rate:",(residuals[-1]/residuals[0])**(1.0/len(residuals))
            assert( (residuals[-1]/residuals[0])**(1.0/len(residuals)) < 0.95 )

