"""Test enlarged Krylov (ECG) methods."""
import numpy as np
from numpy.testing import TestCase, assert_allclose, assert_equal

from pyamg.gallery import poisson
from pyamg.krylov import cg, ekcg, srecg_bcgs, srecg_orthodir, srecg_orthodir_new

import pytest


class TestEnlargedCG(TestCase):
    def setUp(self):
        self.A = poisson((16, 16), format='csr')
        self.b = np.ones(self.A.shape[0])
        self.tol = 1e-8
        self.methods = [ekcg, srecg_bcgs, srecg_orthodir]

    def check_t_sweep(self, method):
        A, b, tol = self.A, self.b, self.tol
        iters = []
        for t in [1, 2, 4, 8]:
            res = []
            x, info = method(A, b, t=t, tol=tol, residuals=res)
            msg = f'Problem in {method.__name__} with t={t}.'
            assert_equal(info, 0, err_msg=msg)
            assert np.linalg.norm(b - A @ x) <= 1.01 * tol * np.linalg.norm(b), msg
            iters.append(len(res) - 1)
        assert iters == sorted(iters, reverse=True), \
            f'{method.__name__} iterations increase with t: {iters}'

    def test_baseline(self):
        """Check that EKCG matches CG for t=1."""
        res_cg = []
        cg(self.A, self.b, tol=self.tol, residuals=res_cg)
        for method in [*self.methods, srecg_orthodir_new]:
            res = []
            _, info = method(self.A, self.b, t=1, tol=self.tol, residuals=res)
            assert_equal(info, 0, err_msg=f'Problem in {method.__name__}.')
            assert_allclose(res, res_cg, rtol=1e-6,
                            err_msg=f'Problem in {method.__name__}.')

    def test_iterations_decrease_with_t(self):
        """Increasing t decreases iterations."""
        for method in self.methods:
            self.check_t_sweep(method)

    @pytest.mark.xfail(reason='srecg_orthodir_new breaks down (NaN) for t > 1',
                       strict=True)
    def test_iterations_decrease_with_t_orthodir_new(self):
        with np.errstate(invalid='ignore'):
            self.check_t_sweep(srecg_orthodir_new)

    def test_finite_termination(self):
        """Size n problem converges in at most n/t iterations."""
        A = poisson((4, 4), format='csr')
        b = np.ones(A.shape[0])
        t = 4
        for method in self.methods:
            res = []
            _, info = method(A, b, t=t, tol=self.tol, residuals=res)
            assert_equal(info, 0, err_msg=f'Problem in {method.__name__}.')
            assert len(res) - 1 <= A.shape[0] // t, \
                f'{method.__name__} took {len(res) - 1} iterations'
