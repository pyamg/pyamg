import numpy as np
from scipy.sparse import csr_matrix
from pyamg.gallery import poisson, load_example
from pyamg.classical.cr import binormalize, CR

from numpy.testing import TestCase


class TestCR(TestCase):
    def setUp(self):
        self.cases = []

        #

        # Random matrices, cases 0-2
        np.random.seed(0)
        for N in [2, 3, 5]:
            self.cases.append(csr_matrix(np.random.rand(N, N)))

        # Poisson problems in 1D, cases 3-9
        for N in [2, 3, 5, 7, 10, 11, 19]:
            self.cases.append(poisson((N,), format='csr'))

        # Poisson problems in 2D, cases 10-15
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

        # 1d-tests, should be alternating aggregates
        #       (n-1)/2 < = sum <= (n+1)/2.
        # Test auto thetacs and set thetacs values
        for i in range(3, 10):
            A = self.cases[i]
            h_split_auto = CR(A, method='habituated', thetacr=0.7,
                              thetacs='auto')
            assert(h_split_auto.sum() <= (h_split_auto.shape[0]+1)/2)
            assert(h_split_auto.sum() >= (h_split_auto.shape[0]-1)/2)

            c_split_auto = CR(A, method='concurrent', thetacr=0.7,
                              thetacs='auto')
            assert(c_split_auto.sum() <= (c_split_auto.shape[0]+1)/2)
            assert(c_split_auto.sum() >= (c_split_auto.shape[0]-1)/2)

            h_split = CR(A, method='habituated', thetacr=0.7,
                         thetacs=[0.3, 0.5])
            assert(h_split.sum() <= (h_split.shape[0]+1)/2)
            assert(h_split.sum() >= (h_split.shape[0]-1)/2)

            c_split = CR(A, method='concurrent', thetacr=0.7,
                         thetacs=[0.3, 0.5])
            assert(c_split.sum() <= (c_split.shape[0]+1)/2)
            assert(c_split.sum() >= (c_split.shape[0]-1)/2)

        # 2d-tests. CR is a little more picky with parameters and relaxation
        # type in 2d. Can still bound above by (n+1)/2.
        # Need looser lower bound,
        # say (n+1)/4.
        for i in range(10, 15):
            A = self.cases[i]
            h_split_auto = CR(A, method='habituated', thetacr=0.7,
                              thetacs='auto')
            assert(h_split_auto.sum() <= (h_split_auto.shape[0]+1)/2)
            assert(h_split_auto.sum() >= (h_split_auto.shape[0]-1)/4)

            c_split_auto = CR(A, method='concurrent', thetacr=0.7,
                              thetacs='auto')
            assert(c_split_auto.sum() <= (c_split_auto.shape[0]+1)/2)
            assert(c_split_auto.sum() >= (c_split_auto.shape[0]-1)/4)

            h_split = CR(A, method='habituated', thetacr=0.7,
                         thetacs=[0.3, 0.5])
            assert(h_split.sum() <= (h_split.shape[0]+1)/2)
            assert(h_split.sum() >= (h_split.shape[0]-1)/4)

            c_split = CR(A, method='concurrent', thetacr=0.7,
                         thetacs=[0.3, 0.5])
            assert(c_split.sum() <= (c_split.shape[0]+1)/2)
            assert(c_split.sum() >= (c_split.shape[0]-1)/4)
