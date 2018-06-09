import bind_examples as g
import numpy as np
from numpy.testing import TestCase


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
        assert x[0] == 7.7
        assert J[0] == 7

    def test_9(self):
        J = np.array([3, 2, 1], dtype=np.intc)
        x = np.array([1.0, 2.0, 3.0], dtype=np.double)
        y = np.ones((3,)) + np.ones((3,)) * 1j

        assert g.test9(J, x, y) == 3
        assert x[0] == 7.7
        assert y[0].real == 7.7
        assert y[0].imag == 8.8
