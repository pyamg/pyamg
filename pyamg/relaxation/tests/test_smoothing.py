from pyamg.testing import *

from pyamg.gallery    import poisson
from pyamg import smoothed_aggregation_solver
from pyamg.util.utils import profile_solver
from pyamg.relaxation.smoothing import change_smoothers

methods = ['gauss_seidel',
           'jacobi',
           'richardson',
           'sor',
           'chebyshev',
           'kaczmarz_gauss_seidel',
           'kaczmarz_jacobi',
           'kaczmarz_richardson']

methods2 = [('gauss_seidel', 'richardson'),
            ('gauss_seidel', 'jacobi'),
            ('chebyshev', 'sor'),
            (['gauss_seidel', 'chebyshev'], ['sor', 'jacobi']),
            ('kaczmarz_gauss_seidel', 'kaczmarz_jacobi'),
            (['kaczmarz_gauss_seidel', 'kaczmarz_richardson'], 'kaczmarz_jacobi'),
            ('cgnr', 'cgne'),
            ( ('gauss_seidel', {'iterations' : 3}), None),
            ( [('kaczmarz_gauss_seidel', {'iterations' : 2}), ('gmres', {'maxiter' : 3})], None),
            ( None, ['cg', 'cgnr', 'cgne']) ]
    
class TestSmoothing(TestCase):
    
    def test_solver_parameters(self):
        A = poisson((50,50), format='csr')

        for method in methods:
            #method = ('richardson', {'omega':4.0/3.0})
            ml = smoothed_aggregation_solver(A, presmoother=method, postsmoother=method, max_coarse=10)

            residuals = profile_solver(ml)
            #print "method",method
            #print "residuals",residuals
            #print "convergence rate:",(residuals[-1]/residuals[0])**(1.0/len(residuals))
            assert( (residuals[-1]/residuals[0])**(1.0/len(residuals)) < 0.95 )

        for method in methods2:
            ml = smoothed_aggregation_solver(A, max_coarse=10)
            change_smoothers(ml, presmoother=method[0], postsmoother=method[1])

            residuals = profile_solver(ml)
            #print "method",method
            #print "residuals",residuals
            #print "convergence rate:",(residuals[-1]/residuals[0])**(1.0/len(residuals))
            assert( (residuals[-1]/residuals[0])**(1.0/len(residuals)) < 0.95 )

