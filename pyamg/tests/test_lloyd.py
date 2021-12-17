import pytest
import pyamg
from pyamg import amg_core
import numpy as np
from scipy import sparse
from numpy.testing import assert_array_equal

@pytest.fixture()
def construct_1dfd_graph():
    u = np.ones(9, dtype=np.float64)
    G = np.diag(2 * u, k=0) + np.diag(u[1:], k=1) + np.diag(u[1:], k=-1)
    return G

def test_balanced_lloyd(construct_1dfd_graph):
    G = construct_1dfd_graph

    # one pass
    centers = np.array([1, 7, 8])
    m, centers = pyamg.graph.balanced_lloyd_cluster(G, centers, maxiter=1, rebalance_iters=0)
    assert_array_equal(m, [0, 0, 0, 0, 1, 1, 1, 1, 2])
    assert_array_equal(centers, [1, 5, 8])

    # multiple passes
    centers = np.array([1, 7, 8])
    m, centers = pyamg.graph.balanced_lloyd_cluster(G, centers, maxiter=5, rebalance_iters=0)
    assert_array_equal(m, [0, 0, 0, 0, 1, 1, 1, 2, 2])
    assert_array_equal(centers, [1, 5, 8])

def test_balanced_lloyd_bystep(construct_1dfd_graph):
    G = construct_1dfd_graph
    G = sparse.csr_matrix(G)
    centers = np.array([1, 7, 8], dtype=np.int32)

    # Balanced Initialization
    n = G.shape[0]
    num_clusters = len(centers)

    maxsize = int(4*np.ceil((n / num_clusters)))
    Cptr = np.empty(num_clusters, dtype=np.int32)
    D = np.empty((maxsize, maxsize), dtype=G.dtype)
    P = np.empty((maxsize, maxsize), dtype=np.int32)
    CC = np.empty(n, dtype=np.int32)
    L = np.empty(n, dtype=np.int32)
    q = np.empty(maxsize, dtype=G.dtype)

    m = np.full(n, -1, dtype=np.int32)
    d = np.full(n, np.inf, dtype=G.dtype)
    p = np.full(n, -1, dtype=np.int32)
    pc = np.zeros(n, dtype=np.int32)
    s = np.full(num_clusters, 1, dtype=np.int32)

    for a in range(centers.shape[0]):
        d[centers[a]] = 0
        m[centers[a]] = a

    # >>Check Balanced Initialization
    assert_array_equal(d, [np.inf, 0, np.inf, np.inf, np.inf, np.inf, np.inf, 0, 0])
    assert_array_equal(m, [-1,     0,     -1,     -1,     -1,     -1,     -1, 1, 2])

    # Pass 0 bellman_ford_balanced
    Ap = G.indptr
    Aj = G.indices
    Ax = G.data
    changed = amg_core.bellman_ford_balanced(n, Ap, Aj, Ax, centers,
                                             d,  m, p, pc, s,
                                             False);

    # >>Check Pass 0 bellman_ford_balanced
    assert_array_equal(m, [0, 0, 0, 0, 1, 1, 1, 1, 2])
    assert_array_equal(d, [1, 0, 1, 2, 3, 2, 1, 0, 0])
    assert_array_equal(p, [1, -1, 1, 2, 5, 6, 7, -1, -1])
    assert_array_equal(pc, [0, 2, 1, 0, 0, 1, 1, 1, 0])
    assert_array_equal(s, [4, 4, 1])
    assert changed

    # Pass 0 center_nodes
    changed = amg_core.center_nodes(n, Ap, Aj, Ax,
                                    Cptr,
                                    D.ravel(), P.ravel(), CC, L, q,
                                    centers, d, m, p, s);

    # >>Check Pass 0 center_nodes
    assert_array_equal(centers, [1, 5, 8])
    assert_array_equal(m, [0, 0, 0, 0, 1, 1, 1, 1, 2])
    assert_array_equal(d, [1, 0, 1, 2, 1, 0, 1, 2, 0])
    # tareq: assert_array_equal(p, [1, -1, 1, 2, 5, -1, 5, 6, -1])
    assert_array_equal(p, [1, -1, 1, 2, 5, 5, 5, 6, -1])
    # tareq: assert_array_equal(pc, [0, 2, 1, 0, 0, 2, 1, 0, 0])
    assert_array_equal(pc, [0, 2, 1, 0, 0, 1, 1, 1, 0])
    assert_array_equal(s, [4, 4, 1])
    assert changed

    # Pass 1 bellman_ford_balanced
    changed = amg_core.bellman_ford_balanced(n, Ap, Aj, Ax, centers,
                                             d,  m, p, pc, s,
                                             False);

    # >>Check Pass 1 bellman_ford_balanced
    assert_array_equal(m, [0, 0, 0, 0, 1, 1, 1, 2, 2])
    assert_array_equal(d, [1, 0, 1, 2, 1, 0, 1, 1, 0])
    # tareq: assert_array_equal(p, [1, -1, 1, 2, 5, -1, 5, 8, -1])
    assert_array_equal(p, [1, -1, 1, 2, 5, 5, 5, 8, -1])
    # tareq: assert_array_equal(pc, [0, 2, 1, 0, 0, 2, 0, 0, 1])
    assert_array_equal(pc, [0, 2, 1, 0, 0, 1, 0, 1, 1])
    assert_array_equal(s, [4, 3, 2])
    assert changed

    # Pass 1 center_nodes
    changed = amg_core.center_nodes(n, Ap, Aj, Ax,
                                    Cptr,
                                    D.ravel(), P.ravel(), CC, L, q,
                                    centers, d, m, p, s);

    # >>Check Pass 1 center_nodes
    assert_array_equal(centers, [1, 5, 8])
    assert_array_equal(m, [0, 0, 0, 0, 1, 1, 1, 2, 2])
    assert_array_equal(d, [1, 0, 1, 2, 1, 0, 1, 1, 0])
    # tareq: assert_array_equal(p, [1, -1, 1, 2, 5, -1, 5, 8, -1])
    assert_array_equal(p, [1, -1, 1, 2, 5, 5, 5, 8, -1])
    # tareq: assert_array_equal(pc, [0, 2, 1, 0, 0, 2, 0, 0, 1])
    assert_array_equal(pc, [0, 2, 1, 0, 0, 1, 0, 1, 1])
    assert_array_equal(s, [4, 3, 2])
    assert not changed

