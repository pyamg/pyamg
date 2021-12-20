"""Test binding."""

import numpy as np
from numpy.testing import TestCase
import pytest

import pyamg.amg_core.tests.bind_examples as g


class TestDocstrings(TestCase):
    def test_1(self):
        assert g.test1.__doc__.strip() == 'Testing docstring'
        assert g.test1(1) == 1

    def test_2(self):
        assert g.test1.__doc__.strip() == 'Testing docstring'
        assert g.test2(1) == 1

    def test_3(self):
        assert g.test1.__doc__.strip() == 'Testing docstring'
        assert g.test3(1) == 1

    def test_4(self):
        assert g.test1.__doc__.strip() == 'Testing docstring'
        assert g.test4(1) == 1


class TestUntemplated(TestCase):
    def test_5(self):
        assert g.test5(1) == 1

    def test_6(self):
        assert g.test6(1) == 1

    def test_7(self):
        assert g.test7(1) == 1


class TestVectors(TestCase):
    def test_8(self):
        n = 1
        m = 2
        x = np.array([1.0, 2.0, 3.0], dtype=np.double)
        J = np.array([1, 2, 3], dtype=np.intc)

        assert g.test8(n, m, x, J) == 1
        assert x[0] == 7.5
        assert J[0] == 7

    def test_9(self):
        J = np.array([3, 2, 1], dtype=np.intc)
        x = np.array([1.0, 2.0, 3.0], dtype=np.double)
        y = np.ones((3,)) + np.ones((3,)) * 1j

        assert g.test9(J, x, y) == 3
        assert x[0] == 7.5
        assert y[0].real == 7.5
        assert y[0].imag == 8.25

    def test_10a(self):
        # int32, float32
        J = np.array([1, 1, 1], dtype=np.int32)
        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        assert g.test10(J, x) == 1
        assert J[0] == 0
        assert x[0] == 7.5

    def test_10b(self):
        # bool, float32
        J = np.array([1, 1, 1], dtype=bool)
        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        assert g.test10(J, x) == 1
        assert J[0] == 0
        assert x[0] == 7.5

    def test_10c(self):
        # int32, double
        J = np.array([1, 1, 1], dtype=np.int32)
        x = np.array([1.0, 2.0, 3.0], dtype=np.float64)

        assert g.test10(J, x) == 1
        assert J[0] == 0
        assert x[0] == 7.5

    def test_10d(self):
        # int32, complex float
        J = np.array([1, 1, 1], dtype=np.int32)
        x = np.array([1.0, 2.0, 3.0], dtype=np.complex64)

        assert g.test10(J, x) == 1
        assert J[0] == 0
        assert x[0].real == 7.5

    def test_10e(self):
        # int32, complex double
        J = np.array([1, 1, 1], dtype=np.int32)
        x = np.array([1.0, 2.0, 3.0], dtype=np.complex128)

        assert g.test10(J, x) == 1
        assert J[0] == 0
        assert x[0].real == 7.5

    def test_10f(self):
        # int8, float32  (should FAIL on upconvert)
        J = np.array([1, 1, 1], dtype=np.int8)
        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        pytest.raises(TypeError, g.test10, J, x)

    def test_10g(self):
        # int64, float32  (should FAIL on downconvert)
        J = np.array([1, 1, 1], dtype=np.int64)
        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        pytest.raises(TypeError, g.test10, J, x)

    def test_10h(self):
        # int32, float16  (should FAIL on upconvert)
        J = np.array([1, 1, 1], dtype=np.int32)
        x = np.array([1.0, 2.0, 3.0], dtype=np.float16)

        pytest.raises(TypeError, g.test10, J, x)

    def test_10i(self):
        # int64, float32  (should FAIL on downconvert)
        J = np.array([1, 1, 1], dtype=np.int64)
        x = np.array([1.0, 2.0, 3.0], dtype=np.float64)

        pytest.raises(TypeError, g.test10, J, x)
