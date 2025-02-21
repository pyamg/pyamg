"""Test Floyd-Warshall."""
import numpy as np
from numpy.testing import TestCase
from scipy import sparse

from pyamg import amg_core


class TestAllPairs(TestCase):
    def setUp(self):
        cases = []

        # https://www.geeksforgeeks.org/floyd-warshall-algorithm-dp-16/
        A = np.zeros((4, 4))
        A[0, 3] = 10
        A[2, 3] = 1
        A[1, 2] = 3
        A[0, 1] = 5
        A = sparse.csr_array(A)

        inf = np.inf
        D_ref = np.array([[0,  5.,  8.,  9.],
                          [inf, 0,  3.,  4.],
                          [inf, inf, 0,  1.],
                          [inf, inf, inf, 0]])
        P_ref = np.array([[0,  0,  1, 2],
                          [-1,  1,  1, 2],
                          [-1, -1,  2, 2],
                          [-1, -1, -1, 3]], dtype=np.int32)
        cases.append([A, D_ref, P_ref])

        # https://github.com/epomp447/Floyd-Warshall-Algorithm-Java-
        A = np.zeros((5, 5))
        A[0, [1, 2, 4]] = [3, 8, -4]
        A[1, [3, 4]] = [1, 7]
        A[2, 1] = 4
        A[3, [0, 2]] = [2, -5]
        A[4, 3] = 6
        A = sparse.csr_array(A)

        D_ref = np.array([[0.,  1., -3.,  2., -4.],
                          [3.,  0., -4.,  1., -1.],
                          [7.,  4.,  0.,  5.,  3.],
                          [2., -1., -5.,  0., -2.],
                          [8.,  5.,  1.,  6.,  0.]])
        P_ref = np.array([[0, 2, 3, 4, 0],
                          [3, 1, 3, 1, 0],
                          [3, 2, 2, 1, 0],
                          [3, 2, 3, 3, 0],
                          [3, 2, 3, 4, 4]], dtype=np.int32)
        cases.append([A, D_ref, P_ref])

        # https://en.wikipedia.org/wiki/Floyd%E2%80%93Warshall_algorithm
        A = np.zeros((4, 4))
        A[0, 2] = -2
        A[2, 3] = 2
        A[3, 1] = -1
        A[1, 0] = 4
        A[1, 2] = 3
        A = sparse.csr_array(A)

        D_ref = np.array([[0., -1, -2, 0],
                          [4,  0,  2, 4],
                          [5,  1,  0, 2],
                          [3, -1,  1, 0]])
        P_ref = np.array([[0, 3, 0, 2],
                          [1, 1, 0, 2],
                          [1, 3, 2, 2],
                          [1, 3, 0, 3]], dtype=np.int32)
        cases.append([A, D_ref, P_ref])

        self.cases = cases

    def test_floyd_warshall(self):
        """Test Floyd-Warshall."""
        for A, D_ref, P_ref in self.cases:
            num_nodes = A.shape[0]
            N = num_nodes
            D = np.full((N, N), np.inf, dtype=A.dtype)
            P = np.full((N, N), -1, dtype=np.int32)
            C = np.arange(0, N, dtype=np.int32)
            L = np.arange(0, N, dtype=np.int32)
            m = np.ones((N,), dtype=np.int32)
            a = 1

            amg_core.floyd_warshall(num_nodes, A.indptr, A.indices, A.data,
                                    D.ravel(), P.ravel(), C, L, m, a, N)

            np.testing.assert_array_equal(D, D_ref)
            np.testing.assert_array_equal(P, P_ref)
