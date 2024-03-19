"""Test scipy methods."""
import inspect
from functools import partial

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
        def cb(x, normb):
            scipyres.append(x*normb)

        for case in self.cases:
            A = case['A']
            b = case['b']
            x0 = case['x0']
            tol = case['tol']

            kwargs = dict(tol=tol, restart=3, maxiter=2)

            mgsres = []
            _ = gmres_mgs(A, b, x0, residuals=mgsres, **kwargs)

            hhres = []
            _ = gmres_householder(A, b, x0, residuals=hhres, **kwargs)

            scipyres = []
            normb = np.linalg.norm(b)
            callback = partial(cb, normb=normb)

            # check if scipy gmres has rtol
            kwargs['atol'] = 0
            if 'rtol' in inspect.getfullargspec(sla.gmres).args:
                kwargs['rtol'] = kwargs.pop(tol)

            _ = sla.gmres(A, b, x0, callback=callback, callback_type='pr_norm', **kwargs)

            assert_array_almost_equal(mgsres[1:], scipyres)
            assert_array_almost_equal(hhres[1:], scipyres)
