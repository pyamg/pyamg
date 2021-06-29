"""Test clustering routines in amg_core."""
import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt
from aggviz import plotaggs
import pyamg

import pyamg.amg_core as amg_core
from pyamg.gallery import load_example

data = load_example('unit_square')
A = data['A'].tocsr()
E = data['elements']
V = data['vertices']

A.data[:] = 1.0
np.random.seed(625)
c = np.int32(np.random.permutation(A.shape[0]-1)[:25])
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
print(m)
print(c)

AggOp = sparse.coo_matrix((np.ones(len(m)), (np.arange(len(m)),m))).tocsr()

fig, ax = plt.subplots(figsize=(10,10))
ax.triplot(V[:,0], V[:,1], E, color='0.5', lw=1.0)
ax.plot(V[:,0],V[:,1], 's', ms=5, markeredgecolor='w', color='tab:gray')

ax.plot(V[c,0],V[c,1], '*', ms=30, markeredgecolor='w', color='tab:green')

for k, agg in enumerate(AggOp.T):                                    # for each aggregate
    #aggids = agg.indices                               # get the indices
    aggids = np.where(m == k)[0]
    print(k, aggids, c[k])
    for i in aggids:
        for j in aggids:
            if A[i,j]:
                plt.plot([V[i,0], V[j,0]], [V[i,1], V[j,1]], 'k-', lw=3)
    plt.plot(V[aggids,0], V[aggids,1], 'o', ms=8)

for i in range(n):
    plt.text(V[i,0], V[i,1], f'{m[i]}')

for j in range(len(c)):
    i = c[j]
    plt.text(V[i,0], V[i,1], f'{i}', color='m')
plt.show()
