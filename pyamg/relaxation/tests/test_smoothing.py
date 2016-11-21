from pyamg.gallery import poisson
from pyamg import smoothed_aggregation_solver
from pyamg.util.utils import profile_solver
from pyamg.relaxation.smoothing import change_smoothers

from numpy.testing import TestCase

methods = [('gauss_seidel', {'sweep': 'symmetric'}),
           'jacobi',
           'richardson',
           ('sor', {'sweep': 'symmetric'}),
           'chebyshev',
           ('gauss_seidel_ne', {'sweep': 'symmetric'}),
           'jacobi_ne',
           ('gauss_seidel_nr', {'sweep': 'symmetric'}),
           ('schwarz', {'sweep': 'symmetric'}),
           ('strength_based_schwarz', {'sweep': 'symmetric'})]

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

# Symmetric smoothing schemes
methods3 = [[[('gauss_seidel', {'sweep': 'forward'}), None],
             [('gauss_seidel', {'sweep': 'backward'}), None]],
            [[('gauss_seidel_nr', {'sweep': 'backward'}), 'jacobi'],
             [('gauss_seidel_nr', {'sweep': 'forward'}), 'jacobi']],
            [[('jacobi', {'iterations': 2}), ('jacobi', {'iterations': 1})],
             [('jacobi', {'iterations': 2}), ('jacobi', {'iterations': 1})]],
            [[('gauss_seidel_ne', {'sweep': 'forward'}), None],
             [('gauss_seidel_ne', {'sweep': 'backward'}), None]],
            [[('block_gauss_seidel', {'sweep': 'backward'}), 'richardson'],
             [('block_gauss_seidel', {'sweep': 'forward'}), 'richardson']],
            [[('jacobi_ne', {'iterations': 2}),
              ('block_jacobi', {'iterations': 1})],
             [('jacobi_ne', {'iterations': 2}),
              ('block_jacobi', {'iterations': 1})]]]

# Non-symmetric smoothing schemes
methods4 = [[[('gauss_seidel', {'sweep': 'forward'}), None],
             [('gauss_seidel', {'sweep': 'forward'}), None]],
            [[('gauss_seidel_nr', {'sweep': 'symmetric'}), 'jacobi'],
             [('gauss_seidel_nr', {'sweep': 'backward'}), 'jacobi']],
            [[('jacobi', {'iterations': 2}),
              ('richardson', {'iterations': 1})],
             [('jacobi', {'iterations': 2}),
              ('richardson', {'iterations': 2})]],
            [[('gauss_seidel_ne', {'sweep': 'backward'}), None],
             [('gauss_seidel_ne', {'sweep': 'backward'}), None]],
            [[('block_gauss_seidel', {'sweep': 'backward'}),
              ('jacobi', {'iterations': 1})],
             [('block_gauss_seidel', {'sweep': 'forward'}),
              ('jacobi', {'iterations': 2})]],
            [[('jacobi_ne', {'iterations': 1}),
              ('block_jacobi', {'iterations': 1})],
             [('jacobi_ne', {'iterations': 2}),
              ('block_jacobi', {'iterations': 1})]]]


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
            assert(ml.symmetric_smoothing)

        for method in methods2:
            ml = smoothed_aggregation_solver(A, max_coarse=10)
            change_smoothers(ml, presmoother=method[0], postsmoother=method[1])

            residuals = profile_solver(ml)
            assert((residuals[-1]/residuals[0])**(1.0/len(residuals)) < 0.95)
            assert(not ml.symmetric_smoothing)

        for method in methods3:
            ml = smoothed_aggregation_solver(A, max_coarse=10)
            change_smoothers(ml, presmoother=method[0], postsmoother=method[1])
            assert(ml.symmetric_smoothing)

        for method in methods4:
            ml = smoothed_aggregation_solver(A, max_coarse=10)
            change_smoothers(ml, presmoother=method[0], postsmoother=method[1])
            assert(not ml.symmetric_smoothing)
