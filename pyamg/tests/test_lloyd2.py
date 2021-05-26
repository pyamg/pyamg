"""Test clustering routines in amg_core."""
import numpy as np
import pyamg.amg_core as amg_core
from pyamg.gallery import load_example
import scipy.sparse as sparse

from numpy.testing import TestCase, assert_equal, assert_array_equal


A = load_example('unit_square')['A'].tocsr()
A.data[:] = 1.0
c = np.array([0, 10, 20, 30], dtype=np.int32)

n = A.shape[0]
num_nodes = A.shape[0]

Cptr = np.array([0, 1], dtype=np.int32)
D = np.zeros((n, n), dtype=A.dtype)
P = np.zeros((n, n), dtype=np.int32)
C  = np.arange(0, n, dtype=np.int32)
L  = np.arange(0, n, dtype=np.int32)

q = np.zeros((n, n), dtype=A.dtype)
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
