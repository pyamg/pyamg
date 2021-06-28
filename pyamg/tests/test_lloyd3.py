"""Test clustering routines in amg_core."""
import numpy as np
import pyamg.amg_core as amg_core
from pyamg.gallery import load_example
import scipy.sparse as sparse

from numpy.testing import TestCase, assert_equal, assert_array_equal

A = load_example('unit_square')['A'].tocsr()
A.data[:] = 1.0
#c = np.array([0, 10, 20, 30,  40,  50, 60, 70, 80, 90, 100, 110, 120, 130,  140, 150, 160, 170, 180], dtype=np.int32)
c = np.random.randint(0,A.shape[0]-1,20, dtype=np.int32)
print(c)

n = A.shape[0]
num_nodes = A.shape[0]

Cptr = np.empty(len(c), dtype=np.int32)
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

amg_core.lloyd_cluster_balanced(num_nodes,
                                A.indptr, A.indices, A.data,
                                Cptr, D.ravel(), P.ravel(), C, L,
                                q.ravel(),
                                c, d, m, p,
                                pc, s,
                                True)
