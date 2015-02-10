from pyamg.gallery import poisson
from pyamg import smoothed_aggregation_solver
from pyamg.util.utils import profile_solver
from pyamg.relaxation.smoothing import change_smoothers

from numpy.testing import TestCase

methods = ['gauss_seidel',
           'jacobi',
           'richardson',
           'sor',
           'chebyshev',
           'gauss_seidel_ne',
           'jacobi_ne',
           'gauss_seidel_nr',
           'schwarz',
           'strength_based_schwarz']

methods2 = [('gauss_seidel', 'richardson'),
            ('gauss_seidel', 'jacobi'),
            ('chebyshev', 'sor'),
            (['gauss_seidel', 'chebyshev'], ['sor', 'jacobi']),
            ('gauss_seidel_ne', 'jacobi_ne'),
            (['gauss_seidel_ne', 'gauss_seidel_nr'], 'jacobi_ne'),
            ('cgnr', 'cgne'),
            ('schwarz', 'strength_based_schwarz'),
            (('gauss_seidel', {'iterations': 3}), None),
            ([('gauss_seidel_ne', {'iterations': 2}),
              ('gmres', {'maxiter': 3})], None),
            (None, ['cg', 'cgnr', 'cgne'])]


class TestSmoothing(TestCase):
    def test_solver_parameters(self):
        A = poisson((50, 50), format='csr')

        for method in methods:
            # method = ('richardson', {'omega':4.0/3.0})
            ml = smoothed_aggregation_solver(A, presmoother=method,
                                             postsmoother=method,
                                             max_coarse=10)

            residuals = profile_solver(ml)
            assert((residuals[-1]/residuals[0])**(1.0/len(residuals)) < 0.95)

        for method in methods2:
            ml = smoothed_aggregation_solver(A, max_coarse=10)
            change_smoothers(ml, presmoother=method[0], postsmoother=method[1])

            residuals = profile_solver(ml)
            assert((residuals[-1]/residuals[0])**(1.0/len(residuals)) < 0.95)
