import numpy as np
from pyamg.gallery.diffusion import diffusion_stencil_2d

from numpy.testing import TestCase, assert_equal, assert_almost_equal


class TestDiffusionStencil2D(TestCase):
    def test_simple_finite_difference(self):
        # isotropic
        stencil = [[0.0, -1.0, 0.0],
                   [-1.0, 4.0, -1.0],
                   [0.0, -1.0, 0.0]]
        assert_equal(diffusion_stencil_2d(epsilon=1.0, theta=0.0,
                     type='FD'), stencil)

        # weak horizontal
        stencil = [[0.0, -1.0, 0.0],
                   [-0.5, 3.0, -0.5],
                   [0.0, -1.0, 0.0]]
        assert_almost_equal(diffusion_stencil_2d(epsilon=0.5, theta=0.0,
                            type='FD'), stencil)

        # weak vertical
        stencil = [[0.0, -0.5, 0.0],
                   [-1.0, 3.0, -1.0],
                   [0.0, -0.5, 0.0]]
        assert_almost_equal(diffusion_stencil_2d(epsilon=0.5, theta=np.pi/2,
                            type='FD'), stencil)

    def test_simple_finite_element(self):
        # isotropic
        stencil = np.array([[-1.0, -1.0, -1.0],
                            [-1.0, 8.0, -1.0],
                            [-1.0, -1.0, -1.0]]) / 3.0
        assert_almost_equal(diffusion_stencil_2d(epsilon=1.0, theta=0.0,
                            type='FE'), stencil)

        # weak horizontal
        # assert_almost_equal(diffusion_stencil_2d(epsilon=0.5, theta=0.0,
        # type='FE'), stencil)

        # weak vertical
        # assert_almost_equal(diffusion_stencil_2d(epsilon=0.5, theta=pi/2,
        # type='FE'), stencil)

    def test_zero_sum(self):
        """test that stencil entries sum to zero"""
        for type in ['FD', 'FE']:
            for theta in [np.pi/8, np.pi/5, np.pi/4, np.pi/3, np.pi/2, np.pi]:
                for epsilon in [0.001, 0.01, 1.0]:
                    stencil = diffusion_stencil_2d(epsilon=epsilon,
                                                   theta=theta, type=type)
                    assert_almost_equal(stencil.sum(), 0.0)

    def test_rotation_invariance(self):
        """test invariance to theta when epsilon=1.0"""

        for type in ['FD', 'FE']:
            expected = diffusion_stencil_2d(epsilon=1.0, theta=0.0, type=type)
            for theta in [np.pi/8, np.pi/4, np.pi/3, np.pi/2, np.pi]:
                result = diffusion_stencil_2d(epsilon=1.0,
                                              theta=theta, type=type)
                assert_almost_equal(result, expected)
