import numpy as np
from pyamg.gallery.stencil import stencil_grid

from numpy.testing import TestCase, assert_equal


class TestStencil(TestCase):
    def test_1d(self):
        stencil = np.array([1, 2, 3])

        result = stencil_grid(stencil, (3,)).toarray()
        expected = np.array([[2, 3, 0],
                             [1, 2, 3],
                             [0, 1, 2]])
        assert_equal(result, expected)

        stencil = np.array([1, 2, 3, 4, 5])
        result = stencil_grid(stencil, (5,)).toarray()
        expected = np.array([[3, 4, 5, 0, 0],
                             [2, 3, 4, 5, 0],
                             [1, 2, 3, 4, 5],
                             [0, 1, 2, 3, 4],
                             [0, 0, 1, 2, 3]])
        assert_equal(result, expected)

    def test_2d(self):
        stencil = np.array([[1, 2, 3],
                           [4, 5, 6],
                           [7, 8, 9]])

        result = stencil_grid(stencil, (2, 3)).toarray()
        expected = np.array([[5, 6, 0, 8, 9, 0],
                             [4, 5, 6, 7, 8, 9],
                             [0, 4, 5, 0, 7, 8],
                             [2, 3, 0, 5, 6, 0],
                             [1, 2, 3, 4, 5, 6],
                             [0, 1, 2, 0, 4, 5]])
        assert_equal(result, expected)

    def test_poisson1d(self):
        stencil = np.array([-1, 2, -1])

        cases = []
        cases.append(((1,), np.array([[2]])))
        cases.append(((2,), np.array([[2, -1],
                                      [-1, 2]])))
        cases.append(((4,), np.array([[2, -1, 0, 0],
                                      [-1, 2, -1, 0],
                                      [0, -1, 2, -1],
                                      [0, 0, -1, 2]])))

        for grid, expected in cases:
            result = stencil_grid(stencil, grid).toarray()
            assert_equal(result, expected)

    def test_poisson2d_5pt(self):
        stencil = np.array([[0, -1, 0],
                           [-1, 4, -1],
                           [0, -1, 0]])

        cases = []
        cases.append(((1, 1), np.array([[4]])))
        cases.append(((2, 1), np.array([[4, -1],
                                        [-1, 4]])))
        cases.append(((1, 2), np.array([[4, -1],
                                        [-1, 4]])))
        cases.append(((1, 3), np.array([[4, -1, 0],
                                        [-1, 4, -1],
                                        [0, -1, 4]])))
        cases.append(((2, 2), np.array([[4, -1, -1, 0],
                                        [-1, 4, 0, -1],
                                        [-1, 0, 4, -1],
                                        [0, -1, -1, 4]])))

        for grid, expected in cases:
            result = stencil_grid(stencil, grid).toarray()
            assert_equal(result, expected)

    def test_poisson2d_9pt(self):
        stencil = np.array([[-1, -1, -1],
                            [-1, 8, -1],
                            [-1, -1, -1]])

        cases = []
        cases.append(((1, 1), np.array([[8]])))
        cases.append(((2, 1), np.array([[8, -1],
                                        [-1, 8]])))
        cases.append(((1, 2), np.array([[8, -1],
                                        [-1, 8]])))
        cases.append(((1, 3), np.array([[8, -1, 0],
                                        [-1, 8, -1],
                                        [0, -1, 8]])))
        cases.append(((2, 2), np.array([[8, -1, -1, -1],
                                        [-1, 8, -1, -1],
                                        [-1, -1, 8, -1],
                                        [-1, -1, -1, 8]])))

        for grid, expected in cases:
            result = stencil_grid(stencil, grid).toarray()
            assert_equal(result, expected)

    def test_poisson3d_7pt(self):
        stencil = np.array([[[0, 0, 0],
                             [0, -1, 0],
                             [0, 0, 0]],
                            [[0, -1, 0],
                             [-1, 6, -1],
                             [0, -1, 0]],
                            [[0, 0, 0],
                             [0, -1, 0],
                             [0, 0, 0]]])

        cases = []
        cases.append(((1, 1, 1), np.array([[6]])))
        cases.append(((2, 1, 1), np.array([[6, -1],
                                           [-1, 6]])))
        cases.append(((2, 2, 1), np.array([[6, -1, -1, 0],
                                           [-1, 6, 0, -1],
                                           [-1, 0, 6, -1],
                                           [0, -1, -1, 6]])))
        cases.append(((2, 2, 2), np.array([[6, -1, -1, 0, -1, 0, 0, 0],
                                           [-1, 6, 0, -1, 0, -1, 0, 0],
                                           [-1, 0, 6, -1, 0, 0, -1, 0],
                                           [0, -1, -1, 6, 0, 0, 0, -1],
                                           [-1, 0, 0, 0, 6, -1, -1, 0],
                                           [0, -1, 0, 0, -1, 6, 0, -1],
                                           [0, 0, -1, 0, -1, 0, 6, -1],
                                           [0, 0, 0, -1, 0, -1, -1, 6]])))

        for grid, expected in cases:
            result = stencil_grid(stencil, grid).toarray()
            assert_equal(result, expected)
