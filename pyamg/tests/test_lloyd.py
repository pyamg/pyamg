"""Test clustering routines in amg_core."""
import numpy as np
import pyamg.amg_core as amg_core
import scipy.sparse as sparse

from numpy.testing import TestCase, assert_equal, assert_array_equal

############################################################
# (1) 12 node undirected, unit length
############################################################
#
#         _[1] ---- [7]
#       /   |        |
#     /     |        |
# [0] ---- [2] ---- [6] ---- [8]
#           |        |        |
#           |        |        |
#          [3] ---- [5] ---- [9] _
#                    |        |    \
#                    |        |      \
#                   [4] ---- [10]---- [11]
xy = np.array([[0, 2],
               [1, 3],
               [1, 2],
               [1, 1],
               [2, 0],
               [2, 1],
               [2, 2],
               [2, 3],
               [3, 2],
               [3, 1],
               [3, 0],
               [4, 0]])
A = np.zeros((12, 12))
A[0, [1, 2]] = 1
A[1, [0, 2, 7]] = 1
A[2, [0, 1, 3, 6]] = 1
A[3, [2, 5]] = 1
A[4, [5, 10]] = 1
A[5, [3, 4, 6, 9]] = 1
A[6, [2, 5, 7, 8]] = 1
A[7, [1, 6]] = 1
A[8, [6, 9]] = 1
A[9, [5, 8, 10, 11]] = 1
A[10, [4, 9, 11]] = 1
A[11, [9, 10]] = 1
A[np.arange(12), np.arange(12)] = [2, 3, 4, 2, 2, 4, 4, 2, 2, 4, 3, 2]
A = sparse.csr_matrix(A)
np.random.seed(2244369509)
A.data[:] = np.random.rand(len(A.data)) * 2
n = A.shape[0]
num_nodes = A.shape[0]

Cptr = np.array([0, 1], dtype=np.int32)
D = np.zeros((n, n), dtype=A.dtype)
P = np.zeros((n, n), dtype=np.int32)
C  = np.arange(0, n, dtype=np.int32)
L  = np.arange(0, n, dtype=np.int32)

q = np.zeros((n, n), dtype=A.dtype)
c = np.array([0, 1], dtype=np.int32)
d = np.empty(n, dtype=A.dtype)
m = np.empty(n, dtype=np.int32)
p = np.empty(n, dtype=np.int32)
pc = np.empty(n, dtype=np.int32)
s = np.empty(len(c), dtype=np.int32)

changed = amg_core.bellman_ford_balanced(num_nodes,
                                A.indptr, A.indices, A.data,
                                c, d, m, p,
                                pc, s,
                                True)

changed = amg_core.center_nodes(num_nodes,
                      A.indptr, A.indices, A.data,
                      Cptr,
                      D.ravel(), P.ravel(), C, L,
                      q.ravel(),
                      c, d, m, p,
                      s)
