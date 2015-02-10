from numpy.random import seed, rand
from scipy.sparse import csr_matrix
from pyamg.gallery import poisson, load_example
from pyamg.classical.cr import binormalize, CR

from numpy.testing import TestCase


class TestCR(TestCase):
    def setUp(self):
        self.cases = []

        # random matrices
        seed(0)

        for N in [2, 3, 5]:
            self.cases.append(csr_matrix(rand(N, N)))

        # Poisson problems in 1D and 2D
        for N in [2, 3, 5, 7, 10, 11, 19]:
            self.cases.append(poisson((N,), format='csr'))

        for N in [2, 3, 5, 7, 10, 11]:
            self.cases.append(poisson((N, N), format='csr'))

        for name in ['knot', 'airfoil', 'bar']:
            ex = load_example(name)
            self.cases.append(ex['A'].tocsr())

    def test_binormalize(self):
        for A in self.cases:
            C = binormalize(A)

            alpha = abs(1.0-C.multiply(C).sum(axis=1)).max()

            assert(alpha < 1e-4)

    def test_cr(self):
        A = self.cases[6]
        splitting = CR(A)
        assert(splitting.sum() < splitting.shape[0])
