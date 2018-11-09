import numpy as np
import pyamg.gallery
import pyamg.sparse

from numpy.testing import TestCase, assert_almost_equal

class TestScipy(TestCase):
    def test_matvec(self):

        # initialize a seed
        random.seed(678)

        # real
        A = array([[100.0, 0, 0], [0, 101, 0], [0, 0, 99]]))
        A2 = pyamg.sparse.csr(A)
        u = np.random.rand(A.shape[0])

        assert_array_almost_equal(A * u, A2 * u)

        # complex
        A = array([[100+1.0j, 0, 0],
                   [0, 101-1.0j, 0],
                   [0, 0, 99+9.9j]]))
        A2 = pyamg.sparse.csr(A)
        u = np.random.rand(A.shape[0])
        assert_array_almost_equal(A * u, A2 * u)

        # random
        A = pyamg.gallery.sprand(20, 20, 6 / 20.0)
        A2 = pyamg.sparse.csr(A)
        u = np.random.rand(A.shape[0])
        assert_array_almost_equal(A * u, A2 * u)
