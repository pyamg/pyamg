"""Test Pairwise AMG."""
import numpy as np

from numpy.testing import TestCase
from pyamg.aggregation import pairwise_solver
from pyamg.gallery import poisson, linear_elasticity, load_example


class TestPairwise(TestCase):

    def test_spd(self):
        cases = []
        cases.append(poisson((500,), format='csr'))
        cases.append(poisson((50, 50), format='csr'))
        cases.append(linear_elasticity((7, 7), format='bsr')[0])
        cases.append(load_example('airfoil')['A'].tocsr())

        for A in cases:
            for agg, expected in [(('pairwise', {'theta': 0.25, 'matchings': 2}), 0.75),
                                  (('pairwise', {'theta': 0.10, 'matchings': 2}), 0.75),
                                  (('pairwise', {'theta': 0.25, 'matchings': 1}), 0.75),
                                  (('pairwise', {'theta': 0.10, 'matchings': 1}), 0.75)]:

                np.random.seed(0)  # make tests repeatable
                x = np.random.rand(A.shape[0])
                b = A*np.random.rand(A.shape[0])

                ml = pairwise_solver(A, aggregate=agg, max_coarse=10)

                res = []
                x_sol = ml.solve(b, x0=x, maxiter=20, tol=1e-12,
                                 residuals=res)
                del x_sol

                avg_convergence_ratio = (res[-1]/res[0])**(1.0/len(res))
                assert (avg_convergence_ratio < expected)
