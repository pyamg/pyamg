from numpy import array, pi
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
        assert_almost_equal(diffusion_stencil_2d(epsilon=0.5, theta=pi/2,
                            type='FD'), stencil)

    def test_simple_finite_element(self):
        # isotropic
        stencil = array([[-1.0, -1.0, -1.0],
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
            for theta in [pi/8, pi/5, pi/4, pi/3, pi/2, pi]:
                for epsilon in [0.001, 0.01, 1.0]:
                    stencil = diffusion_stencil_2d(epsilon=epsilon,
                                                   theta=theta, type=type)
                    assert_almost_equal(stencil.sum(), 0.0)

    def test_rotation_invariance(self):
        """test invariance to theta when epsilon=1.0"""

        for type in ['FD', 'FE']:
            expected = diffusion_stencil_2d(epsilon=1.0, theta=0.0, type=type)
            for theta in [pi/8, pi/4, pi/3, pi/2, pi]:
                result = diffusion_stencil_2d(epsilon=1.0,
                                              theta=theta, type=type)
                assert_almost_equal(result, expected)
