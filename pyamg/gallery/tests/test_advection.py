"""Test diffusion example."""
import numpy as np
from pyamg.gallery import advection_2d

from numpy.testing import TestCase, assert_array_almost_equal


class TestAvection2D(TestCase):
    def test_simple_finite_difference(self):
        # pi/4
        A_ref = np.array([[1.41421356,  0.,         -0.70710678, 0.],
                          [-0.70710678, 1.41421356,  0.,         -0.70710678],
                          [0.,          0.,          1.41421356, 0.],
                          [0.,          0.,         -0.70710678, 1.41421356]])
        rhs_ref = np.array([0.70710678, 0., 1.41421356, 0.70710678])
        A, rhs = advection_2d((3, 3), theta=np.pi/4)
        assert_array_almost_equal(A_ref, A.todense(), decimal=5)
        assert_array_almost_equal(rhs_ref, rhs, decimal=5)

        # pi/3
        A_ref = np.array([[1.3660254, 0.,        -0.8660254,  0.],
                          [-0.5,      1.3660254,  0.,        -0.8660254],
                          [0.,        0.,         1.3660254,  0.],
                          [0.,        0.,        -0.5,        1.3660254]])
        rhs_ref = np.array([0.5,  0., 1.3660254, 0.8660254])
        A, rhs = advection_2d((3, 3), theta=np.pi/3)
        assert_array_almost_equal(A_ref, A.todense(), decimal=5)
        assert_array_almost_equal(rhs_ref, rhs, decimal=5)
