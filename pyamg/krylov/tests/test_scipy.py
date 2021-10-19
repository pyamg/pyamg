import numpy as np
import scipy.sparse.linalg as sla
from pyamg.krylov._gmres_mgs import gmres_mgs
import pyamg

from numpy.testing import TestCase, assert_array_almost_equal, assert_equal

class TestScipy(TestCase):
    def setUp(self):
        self.cases = []

        np.random.seed(2937804)
        n = 10
        A = np.random.rand(n,n)
        b = np.random.rand(n)
        x0 = np.random.rand(n)
        self.cases.append({'A': A, 'b': b, 'x0': x0, 'tol': 1e-16})

        n = 20
        A = np.random.rand(n,n)
        b = np.random.rand(n)
        x0 = np.random.rand(n)
        self.cases.append({'A': A, 'b': b, 'x0': x0, 'tol': 1e-16})

    def test_gmres(self):
        for case in self.cases:
            A = case['A']
            b = case['b']
            x0 = case['x0']
            tol = case['tol']
            n = A.shape[0]

            res = []
            _ = gmres_mgs(A, b, x0, residuals=res,
                          tol=tol, restrt=3, maxiter=2)

            scipyres = []
            normb = np.linalg.norm(b)
            def cb(x):
                scipyres.append(x*normb)
            _ = sla.iterative.gmres(A, b,  x0, callback=cb, callback_type='pr_norm',
                                    tol=tol, atol=0, restrt=3, maxiter=2)

            print(res)
            print(scipyres)
            assert_array_almost_equal(res[1:], scipyres)
