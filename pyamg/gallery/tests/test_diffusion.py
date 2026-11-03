"""Test diffusion example."""
import numpy as np
from scipy.sparse.linalg import spsolve
from pyamg.gallery.diffusion import diffusion_stencil_2d, diffusion_variable
from pyamg.gallery import stencil_grid

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
        """Test that stencil entries sum to zero."""
        for type in ['FD', 'FE']:
            for theta in [np.pi/8, np.pi/5, np.pi/4, np.pi/3, np.pi/2, np.pi]:
                for epsilon in [0.001, 0.01, 1.0]:
                    stencil = diffusion_stencil_2d(epsilon=epsilon,
                                                   theta=theta, type=type)
                    assert_almost_equal(stencil.sum(), 0.0)

    def test_rotation_invariance(self):
        """Test invariance to theta when epsilon=1.0."""
        for type in ['FD', 'FE']:
            expected = diffusion_stencil_2d(epsilon=1.0, theta=0.0, type=type)
            for theta in [np.pi/8, np.pi/4, np.pi/3, np.pi/2, np.pi]:
                result = diffusion_stencil_2d(epsilon=1.0,
                                              theta=theta, type=type)
                assert_almost_equal(result, expected)


def _interior(grid):
    """Interior node indices (y varying first)."""
    nx, ny = grid
    ix, iy = np.meshgrid(np.arange(1, nx-1), np.arange(1, ny-1), indexing='ij')
    return (ix*ny + iy).ravel()


class TestDiffusionVariable(TestCase):
    def test_identity_stencil(self):
        """Check kappa=I and the Q1 Laplacian stencil."""
        stencil = np.array([[-1.0, -1.0, -1.0],
                            [-1.0, 8.0, -1.0],
                            [-1.0, -1.0, -1.0]]) / 3.0
        for grid in [(3, 3), (5, 5), (8, 8)]:
            A = diffusion_variable(grid)
            expected = stencil_grid(stencil, grid, format='csr')
            interior = _interior(grid)

            assert_almost_equal(A[interior].toarray(), expected[interior].toarray())

    def test_identity_stencil_unequal_spacing(self):
        """Check kappa=I with hx != hy and the anisotropic stencil."""
        for grid in [(5, 9), (9, 5), (4, 7)]:
            hx = 1 / (grid[0] - 1)
            hy = 1 / (grid[1] - 1)
            stencil = (hy / hx) * diffusion_stencil_2d(epsilon=(hx / hy)**2, type='FE')
            A = diffusion_variable(grid)
            expected = stencil_grid(stencil, grid, format='csr')
            interior = _interior(grid)

            assert_almost_equal(A[interior].toarray(), expected[interior].toarray())

    def test_constant_rotated_anisotropic_stencil(self):
        """Check constant kappa = Q diag(1, eps) Q^T and the anisotropic stencil."""
        grid = (6, 6)
        interior = _interior(grid)
        for epsilon in [0.01, 0.1, 1.0]:
            for theta in [0.0, np.pi/6, np.pi/4, np.pi/2, -np.pi/3]:
                C, S = np.cos(theta), np.sin(theta)
                Q = np.array([[C, -S], [S, C]])
                K = Q @ np.diag([1.0, epsilon]) @ Q.T
                A = diffusion_variable(grid, kappa=lambda x, y, K=K: K)
                stencil = diffusion_stencil_2d(epsilon=epsilon, theta=theta, type='FE')
                expected = stencil_grid(stencil, grid, format='csr')

                assert_almost_equal(A[interior].toarray(), expected[interior].toarray())

    def test_manufactured_solution(self):
        """Test second-order convergence for a manufactured solution.

        u = (1 - cos(2 pi x))(1 - cos(2 pi y)) has u = 0 and grad u = 0 on the
        boundary of [0, 1]^2.

        kappa = [[x+1,   xy/2],
                 [xy/2,  y+2 ]]

            c = 1 + xy
        """
        a = 2 * np.pi

        def kappa(x, y):
            return np.array([[x+1.0, x*y/2],
                             [x*y/2, y+2.0]])

        def c(x, y):
            return 1.0 + x*y

        def f(x, y):
            U   = 1 - np.cos(a*x)
            Ux  = a * np.sin(a*x)
            Uxx = a**2 * np.cos(a*x)
            V   = 1 - np.cos(a*y)
            Vy  = a * np.sin(a*y)
            Vyy = a**2 * np.cos(a*y)
            k11, k12, k22 = 1.0 + x, x * y / 2, 2.0 + y

            divkgradu = (Ux * V
                         + k11 * Uxx * V
                         + (y / 2) * U * Vy
                         + 2 * k12 * Ux * Vy
                         + (x / 2) * Ux * V
                         + U * Vy
                         + k22 * U * Vyy)
            return -divkgradu + c(x, y) * U * V

        def zero(x, y):
            return np.zeros((2, 2))

        def one(x, y):
            return 1.0

        errors = []
        for n in [9, 17, 33, 65]:
            grid = (n, 2 * n - 1)
            A = diffusion_variable(grid, kappa=kappa, c=c, format='csc')
            M = diffusion_variable(grid, kappa=zero, c=one)  # Q1 mass matrix
            X, Y = np.meshgrid(np.linspace(0, 1, grid[0]),
                               np.linspace(0, 1, grid[1]), indexing='ij')
            X, Y = X.ravel(), Y.ravel()
            uh = spsolve(A, M @ f(X, Y))
            u = (1 - np.cos(a*X)) * (1 - np.cos(a*Y))
            errors.append(np.abs(uh - u).max())

        rates = np.log2(np.array(errors[:-1]) / np.array(errors[1:]))
        assert errors[-1] < 5e-3
        assert np.all(rates > 1.9)
