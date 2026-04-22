"""Tests for pyamg.util.mrhs helpers."""
import numpy as np
import pytest
from numpy.testing import (TestCase, assert_array_equal, assert_array_almost_equal,
                           assert_equal)

from pyamg.util.mrhs import (normalize_rhs, restore_shape,
                             per_column_norms, all_converged)


class TestNormalizeRhs(TestCase):
    def test_1d_input(self):
        b = np.arange(5, dtype=float)
        B, shape = normalize_rhs(b)
        assert_equal(B.shape, (5, 1))
        assert_equal(shape, (5,))
        assert B.flags['F_CONTIGUOUS']
        assert_array_equal(B[:, 0], b)

    def test_column_vector_input(self):
        b = np.arange(5, dtype=float).reshape(-1, 1)
        B, shape = normalize_rhs(b)
        assert_equal(B.shape, (5, 1))
        assert_equal(shape, (5, 1))
        assert B.flags['F_CONTIGUOUS']
        assert_array_equal(B, b)

    def test_multi_column_input(self):
        b = np.arange(15, dtype=float).reshape(5, 3)
        B, shape = normalize_rhs(b)
        assert_equal(B.shape, (5, 3))
        assert_equal(shape, (5, 3))
        assert B.flags['F_CONTIGUOUS']
        assert_array_equal(B, b)

    def test_column_slices_are_contiguous(self):
        b = np.arange(15, dtype=float).reshape(5, 3)
        B, _ = normalize_rhs(b)
        for j in range(B.shape[1]):
            assert B[:, j].flags['C_CONTIGUOUS']

    def test_rejects_3d(self):
        b = np.zeros((4, 2, 2))
        with pytest.raises(ValueError, match=r'b\.ndim'):
            normalize_rhs(b)


class TestRestoreShape(TestCase):
    def test_restore_1d(self):
        X = np.arange(5, dtype=float).reshape(5, 1)
        x = restore_shape(X, (5,))
        assert_equal(x.shape, (5,))
        assert_array_equal(x, X[:, 0])

    def test_restore_column_vector(self):
        X = np.arange(5, dtype=float).reshape(5, 1)
        x = restore_shape(X, (5, 1))
        assert_equal(x.shape, (5, 1))
        assert_array_equal(x, X)

    def test_restore_multi_column(self):
        X = np.arange(15, dtype=float).reshape(5, 3)
        x = restore_shape(X, (5, 3))
        assert_equal(x.shape, (5, 3))
        assert_array_equal(x, X)

    def test_roundtrip(self):
        for shape in [(7,), (7, 1), (7, 4)]:
            b = np.random.rand(*shape)
            B, orig = normalize_rhs(b)
            x = restore_shape(B, orig)
            assert_equal(x.shape, shape)
            assert_array_equal(x, b)


class TestPerColumnNorms(TestCase):
    def test_1d(self):
        r = np.array([3.0, 4.0])
        out = per_column_norms(r)
        assert_equal(out.shape, (1,))
        assert_array_almost_equal(out, [5.0])

    def test_2d(self):
        R = np.array([[3.0, 0.0], [4.0, 5.0]])
        out = per_column_norms(R)
        assert_equal(out.shape, (2,))
        assert_array_almost_equal(out, [5.0, 5.0])


class TestAllConverged(TestCase):
    def test_all_below(self):
        normr = np.array([1e-9, 1e-9])
        normb = np.array([1.0, 1.0])
        assert all_converged(normr, 1e-6, normb)

    def test_one_above(self):
        normr = np.array([1e-9, 1e-3])
        normb = np.array([1.0, 1.0])
        assert not all_converged(normr, 1e-6, normb)

    def test_zero_tol(self):
        normr = np.array([0.0, 0.0])
        normb = np.array([1.0, 1.0])
        assert not all_converged(normr, 0.0, normb)
