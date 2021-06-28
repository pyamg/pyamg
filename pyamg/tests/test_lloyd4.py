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
#c = np.array([0, 10, 20, 30,  40,  50, 60, 70, 80, 90, 100, 110, 120, 130,  140, 150, 160, 170, 180], dtype=np.int32)
c = np.random.randint(0,A.shape[0]-1,25, dtype=np.int32)
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

AggOp0 = sparse.coo_matrix((np.ones(len(m)), (np.arange(len(m)),m))).tocsr()

ID = np.kron(np.arange(0, E.shape[0]), np.ones((3,), dtype=np.int32))
V2E = sparse.coo_matrix((np.ones((E.shape[0]*3,), dtype=int), (E.ravel(), ID,)))
G = V2E @ V2E.T

data = pyamg.gallery.load_example('unit_square')
A = data['A'].tocsr()
E = data['elements']
V = data['vertices']
ml = pyamg.smoothed_aggregation_solver(A, keep=True)
AggOp = ml.levels[0].AggOp

fig, ax = plt.subplots(figsize=(10,10))
ax.triplot(V[:,0], V[:,1], E, color='0.5', lw=1.0)
ax.plot(V[:,0],V[:,1], 's', ms=5, markeredgecolor='w', color='tab:gray')

ax.plot(V[c,0],V[c,1], '*', ms=20, markeredgecolor='w', color='tab:green')
#kwargs = {'color': 'tab:blue', 'alpha': 0.4}
#plotaggs(AggOp, V, E, G=A, ax=ax, **kwargs)

kwargs = {'color': 'tab:red', 'alpha': 0.4}
plotaggs(AggOp0, V, E, G=A, ax=ax, **kwargs)

#plt.savefig('aggviz.png')
plt.show()
