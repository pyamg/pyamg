import numpy as np
from scipy import sparse

from pyamg import amg_core

u = np.ones(9, dtype=np.float64)
G = np.diag(2 * u, k=0) + np.diag(u[1:], k=1) + np.diag(u[1:], k=-1)
G = sparse.csr_matrix(G)
centers = np.array([1, 7, 8], dtype=np.int32)

# Balanced Initialization
n = G.shape[0]
num_clusters = len(centers)

maxsize = int(4 * np.ceil((n / num_clusters)))
Cptr = np.zeros(num_clusters, dtype=np.int32)
D = np.empty((maxsize, maxsize), dtype=G.dtype)
P = np.empty((maxsize, maxsize), dtype=np.int32)
CC = np.zeros(n, dtype=np.int32)
L = np.zeros(n, dtype=np.int32)
q = np.zeros(maxsize, dtype=G.dtype)

m = np.full(n, -1, dtype=np.int32)
d = np.full(n, np.inf, dtype=G.dtype)
p = np.full(n, -1, dtype=np.int32)
pc = np.zeros(n, dtype=np.int32)
s = np.full(num_clusters, 1, dtype=np.int32)

for a in range(centers.shape[0]):
    d[centers[a]] = 0
    m[centers[a]] = a

# Pass 0 bellman_ford_balanced
Ap = G.indptr
Aj = G.indices
Ax = G.data
changed = amg_core.bellman_ford_balanced(n, Ap, Aj, Ax, centers,
                                         d, m, p, pc, s,
                                         False)

# Pass 0 center_nodes
print('centers before', centers)
changed = amg_core.center_nodes(n, Ap, Aj, Ax,
                                Cptr,
                                D.ravel(), P.ravel(), CC, L, q,
                                centers, d, m, p, s)
print(changed)
