"""Reference implementations of graph algorithms."""
import numpy as np
import scipy.sparse as sparse
from scipy.sparse import csgraph
from graph_ref import bellman_ford_reference, bellman_ford_balanced_reference
import pyamg
from pyamg.graph import bellman_ford
from pyamg import amg_core

case = 2

if case == 0:
    Edges = np.array([[0, 1],
                      [1, 2],
                      [2, 3],
                      [3, 2],
                      [2, 1],
                      [1, 0]])
    w = np.array([1, 1, 1, 1, 1, 1], dtype=float)
    A = sparse.coo_matrix((w, (Edges[:, 0], Edges[:, 1])))
    A = A.tocsr()
    c = np.array([1,3], dtype=np.int32)


if case == 1:
    Edges = np.array([[1, 4],
                      [3, 1],
                      [1, 3],
                      [0, 1],
                      [0, 2],
                      [3, 2],
                      [1, 2],
                      [4, 3]])
    w = np.array([2, 1, 2, 1, 4, 5, 3, 1], dtype=float)
    A = sparse.coo_matrix((w, (Edges[:, 0], Edges[:, 1])))
    A = A.tocsr()
    c = np.array([0,1,2], dtype=np.int32)

if case == 2:
    data = pyamg.gallery.load_example('unit_square')

    A = data['A'].tocsr()
    A.data = np.abs(A.data)
    n = A.shape[0]
    c = np.array([0,10,20,30], dtype=np.int32)

print(A.toarray())

dpack = {}
print('\nreference BF--')
A = A.tocoo()
d, m, p = bellman_ford_reference(A, c)
print(d, m, p)
dpack['BFref'] = d

print('\npyamg BF--')
A = A.tocsr()
n = A.shape[0]

d = np.empty(n, dtype=A.dtype)
m = np.empty(n, dtype=np.int32)
p = np.empty(n, dtype=np.int32)

amg_core.bellman_ford(n, A.indptr, A.indices, A.data, c, d, m, p, True)
print(d, m, p)

print('\ncsgraph.bellman_ford')
A = A.tocsr()
csgraph.bellman_ford(A, directed=True, indices=c, return_predecessors=True)
print(d.ravel(), p.ravel())
dpack['BF'] = d

print('---------------------')

print('\nreference BFB')
A = A.tocoo()
d, m, p = bellman_ford_balanced_reference(A, c)
print(d, m, p)
dpack['BFBref'] = d

print('\npyamg BFB--')
A = A.tocsr()
n = A.shape[0]

d = np.empty(n, dtype=A.dtype)
m = np.empty(n, dtype=np.int32)
p = np.empty(n, dtype=np.int32)
pc = np.empty(n, dtype=np.int32)
s = np.empty(len(c), dtype=np.int32)

amg_core.bellman_ford_balanced(n, A.indptr, A.indices, A.data, c, d, m, p, pc, s, True)
print(d, m, p)
dpack['BFB'] = d

print('------')
print('difference in BF')
print(dpack['BFref'] - dpack['BF'])
print('difference in BFB')
print(dpack['BFBref'] - dpack['BFB'])
