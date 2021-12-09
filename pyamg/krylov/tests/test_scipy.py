import numpy as np
import scipy.sparse.linalg as sla

from numpy.testing import TestCase, assert_array_almost_equal

from pyamg.krylov._gmres_mgs import gmres_mgs
from pyamg.krylov._gmres_householder import gmres_householder


class TestScipy(TestCase):
    def setUp(self):
        self.cases = []

        np.random.seed(2937804)
        n = 10
        A = np.random.rand(n, n)
        b = np.random.rand(n)
        x0 = np.random.rand(n)
        self.cases.append({'A': A, 'b': b, 'x0': x0, 'tol': 1e-16})

        n = 20
        A = np.random.rand(n, n)
        b = np.random.rand(n)
        x0 = np.random.rand(n)
        self.cases.append({'A': A, 'b': b, 'x0': x0, 'tol': 1e-16})

    def test_gmres(self):
        for case in self.cases:
            A = case['A']
            b = case['b']
            x0 = case['x0']
            tol = case['tol']

            mgsres = []
            _ = gmres_mgs(A, b, x0, residuals=mgsres,
                          tol=tol, restrt=3, maxiter=2)

            hhres = []
            _ = gmres_householder(A, b, x0, residuals=hhres,
                                  tol=tol, restrt=3, maxiter=2)

            scipyres = []
            normb = np.linalg.norm(b)

            def cb(x):
                scipyres.append(x*normb)

            _ = sla.iterative.gmres(A, b, x0, callback=cb, callback_type='pr_norm',
                                    tol=tol, atol=0, restrt=3, maxiter=2)

            assert_array_almost_equal(mgsres[1:], scipyres)
            assert_array_almost_equal(hhres[1:], scipyres)
