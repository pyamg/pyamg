"""Test balanced lloyd clustering."""

import numpy as np
from numpy.testing import assert_array_equal
from scipy import sparse

import pytest

import pyamg
from pyamg import amg_core


@pytest.fixture
def construct_1dfd_graph():
    u = np.ones(9, dtype=np.float64)
    G = np.diag(2 * u, k=0) + np.diag(u[1:], k=1) + np.diag(u[1:], k=-1)
    return G


@pytest.fixture
def construct_graph_laplacian():
    G = np.diag([2, 3, 3, 2, 2, 3, 3, 2], k=0)
    G += np.diag([-1, -1, -1, 0, -1, -1, -1], k=1)
    G += np.diag([-1, -1, -1, 0, -1, -1, -1], k=-1)
    G += np.diag([-1, -1, -1, -1], k=4)
    G += np.diag([-1, -1, -1, -1], k=-4)
    G = np.abs(G.astype(np.float64))
    return G


def _check_pc(p):
    p = np.array(p)
    c = np.bincount(p[p > -1], minlength=len(p))
    return c


def test_balanced_lloyd_1d(construct_1dfd_graph):
    G = construct_1dfd_graph

    # one pass
    centers = np.array([1, 7, 8])
    m, centers = pyamg.graph.balanced_lloyd_cluster(G, centers, maxiter=1,
                                                    rebalance_iters=0)
    assert_array_equal(m, [0, 0, 0, 0, 1, 1, 1, 1, 2])
    assert_array_equal(centers, [1, 5, 8])

    # multiple passes
    centers = np.array([1, 7, 8])
    m, centers = pyamg.graph.balanced_lloyd_cluster(G, centers, maxiter=5,
                                                    rebalance_iters=0)
    assert_array_equal(m, [0, 0, 0, 0, 1, 1, 1, 2, 2])
    assert_array_equal(centers, [1, 5, 8])


def test_balanced_lloyd_1d_bystep(construct_1dfd_graph):
    G = construct_1dfd_graph
    G = sparse.csr_array(G)
    centers = np.array([1, 7, 8], dtype=np.int32)

    # Balanced Initialization
    n = G.shape[0]
    num_clusters = len(centers)

    maxsize = int(4*np.ceil(n / num_clusters))
    Cptr = np.empty(num_clusters, dtype=np.int32)
    D = np.empty((maxsize, maxsize), dtype=G.dtype)
    P = np.empty((maxsize, maxsize), dtype=np.int32)
    CC = np.empty(n, dtype=np.int32)
    L = np.empty(n, dtype=np.int32)
    q = np.empty(maxsize, dtype=G.dtype)

    m = np.full(n, -1, dtype=np.int32)
    d = np.full(n, np.inf, dtype=G.dtype)
    p = np.full(n, -1, dtype=np.int32)
    p[centers] = centers
    pc = np.zeros(n, dtype=np.int32)
    pc[centers] = 1
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
                                             d,  m, p, pc, s, True)

    # >>Check Pass 0 bellman_ford_balanced
    print(m)
    assert_array_equal(m, [0, 0, 0, 0, 1, 1, 1, 1, 2])
    assert_array_equal(d, [1, 0, 1, 2, 3, 2, 1, 0, 0])
    assert_array_equal(p, [1, 1, 1, 2, 5, 6, 7, 7, 8])
    assert_array_equal(pc, [0, 3, 1, 0, 0, 1, 1, 2, 1])
    assert_array_equal(_check_pc(p), pc)
    assert_array_equal(s, [4, 4, 1])
    assert changed

    # Pass 0 center_nodes
    changed = amg_core.center_nodes(n, Ap, Aj, Ax,
                                    Cptr,
                                    D.ravel(), P.ravel(), CC, L, q,
                                    centers, d, m, p, pc, s)

    # >>Check Pass 0 center_nodes
    assert_array_equal(centers, [1, 5, 8])
    assert_array_equal(m, [0, 0, 0, 0, 1, 1, 1, 1, 2])
    assert_array_equal(d, [1, 0, 1, 2, 1, 0, 1, 2, 0])
    assert_array_equal(p, [1, 1, 1, 2, 5, 5, 5, 6, 8])
    assert_array_equal(pc, [0, 3, 1, 0, 0, 3, 1, 0, 1])
    assert_array_equal(_check_pc(p), pc)
    assert_array_equal(s, [4, 4, 1])
    assert changed

    # Pass 1 bellman_ford_balanced
    changed = amg_core.bellman_ford_balanced(n, Ap, Aj, Ax, centers,
                                             d,  m, p, pc, s, True)

    # >>Check Pass 1 bellman_ford_balanced
    assert_array_equal(centers, [1, 5, 8])
    assert_array_equal(m, [0, 0, 0, 0, 1, 1, 1, 2, 2])
    assert_array_equal(d, [1, 0, 1, 2, 1, 0, 1, 1, 0])
    assert_array_equal(p, [1, 1, 1, 2, 5, 5, 5, 8, 8])
    assert_array_equal(pc, [0, 3, 1, 0, 0, 3, 0, 0, 2])
    assert_array_equal(_check_pc(p), pc)
    assert_array_equal(s, [4, 3, 2])
    assert changed

    # Pass 1 center_nodes
    changed = amg_core.center_nodes(n, Ap, Aj, Ax,
                                    Cptr,
                                    D.ravel(), P.ravel(), CC, L, q,
                                    centers, d, m, p, pc, s)

    # >>Check Pass 1 center_nodes
    assert_array_equal(centers, [1, 5, 8])
    assert_array_equal(m, [0, 0, 0, 0, 1, 1, 1, 2, 2])
    assert_array_equal(d, [1, 0, 1, 2, 1, 0, 1, 1, 0])
    assert_array_equal(p, [1, 1, 1, 2, 5, 5, 5, 8, 8])
    assert_array_equal(pc, [0, 3, 1, 0, 0, 3, 0, 0, 2])
    assert_array_equal(_check_pc(p), pc)
    assert_array_equal(s, [4, 3, 2])
    assert not changed


def test_balanced_lloyd_laplacian(construct_graph_laplacian):
    G = construct_graph_laplacian

    # one pass
    centers = np.array([1, 5])
    m, centers = pyamg.graph.balanced_lloyd_cluster(G, centers, maxiter=1,
                                                    rebalance_iters=0)
    assert_array_equal(m, [0, 0, 0, 0, 1, 1, 1, 1])
    assert_array_equal(centers, [1, 5])


def test_balanced_lloyd_laplacian_bystep(construct_graph_laplacian):
    G = construct_graph_laplacian
    G = sparse.csr_array(G)
    centers = np.array([1, 5], dtype=np.int32)

    # Balanced Initialization
    n = G.shape[0]
    num_clusters = len(centers)

    maxsize = int(4*np.ceil(n / num_clusters))
    Cptr = np.empty(num_clusters, dtype=np.int32)
    D = np.empty((maxsize, maxsize), dtype=G.dtype)
    P = np.empty((maxsize, maxsize), dtype=np.int32)
    CC = np.empty(n, dtype=np.int32)
    L = np.empty(n, dtype=np.int32)
    q = np.empty(maxsize, dtype=G.dtype)

    m = np.full(n, -1, dtype=np.int32)
    d = np.full(n, np.inf, dtype=G.dtype)
    p = np.full(n, -1, dtype=np.int32)
    p[centers] = centers
    pc = np.zeros(n, dtype=np.int32)
    pc[centers] = 1
    s = np.full(num_clusters, 1, dtype=np.int32)

    for a in range(centers.shape[0]):
        d[centers[a]] = 0
        m[centers[a]] = a

    # >>Check Balanced Initialization
    assert_array_equal(d, [np.inf, 0, np.inf, np.inf, np.inf, 0, np.inf, np.inf])
    assert_array_equal(m, [-1,     0,     -1,     -1,     -1, 1,     -1,     -1])

    # Pass 0 bellman_ford_balanced
    Ap = G.indptr
    Aj = G.indices
    Ax = G.data
    changed = amg_core.bellman_ford_balanced(n, Ap, Aj, Ax, centers,
                                             d,  m, p, pc, s, True)

    # >>Check Pass 0 bellman_ford_balanced
    assert_array_equal(m, [0, 0, 0, 0, 1, 1, 1, 1])
    assert_array_equal(d, [1, 0, 1, 2, 1, 0, 1, 2])
    assert_array_equal(p, [1, 1, 1, 2, 5, 5, 5, 6])
    assert_array_equal(pc, [0, 3, 1, 0, 0, 3, 1, 0])
    assert_array_equal(_check_pc(p), pc)
    assert_array_equal(s, [4, 4])
    assert changed

    # Pass 0 center_nodes
    changed = amg_core.center_nodes(n, Ap, Aj, Ax,
                                    Cptr,
                                    D.ravel(), P.ravel(), CC, L, q,
                                    centers, d, m, p, pc, s)

    # >>Check Pass 0 center_nodes
    assert_array_equal(centers, [1, 5])
    assert not changed


def test_rebalance():
    #          c        c <- centers
    # 0--1--2--3--4--5--6 <- graph
    # x  x  x  x  x  x  o <- cluster id
    # nearest neigbor unit distance
    G = sparse.diags_array([1., 0, 1], offsets=[-1, 0, 1], shape=(7, 7)).tocsr()
    c = np.array([3, 6], dtype=np.int32)
    m = np.array([0, 0, 0, 0, 0, 0, 1], dtype=np.int32)
    d = np.array([3, 2, 1, 0, 1, 2, 0], dtype=np.float64)
    dist_all = np.empty((2, 40), dtype=np.float64)
    dist_all.fill(np.inf)
    dist_all[0, :36] = np.array([
        0, 1, 2, 3, 4, 5,
        1, 0, 1, 2, 3, 4,
        2, 1, 0, 1, 2, 3,
        3, 2, 1, 0, 1, 2,
        4, 3, 2, 1, 0, 1,
        5, 4, 3, 2, 1, 0], dtype=np.float64)
    dist_all[1, [0]] = np.array([0])

    # elimination penalty
    # cluster 0:
    # 6^2 + 5^2 + 4^2 + 3^2 + 2^2 + 1^2
    # - (3^2 + 2^2 + 1^2 + 0^2 + 1^2 + 2^2)
    # = 72
    # cluster 1:
    # 3**2 - 0**2
    # = 9
    E = pyamg.graph._elimination_penalty(G, m, d, dist_all, num_clusters=2)
    np.testing.assert_array_equal(E, [72, 9])

    # split improvement
    # initial: [3^2 + 2^2 + 1^2 + 0^2 + 1^2 + 2^2, 0^2] = [19, 0]
    # cluster 0 split:
    #     center 1:  - (1^2 + 0^2 + 1^2)
    #     center 4:  - (1^2 + 0^2 + 1^2)
    # S[0] = 19 - 4 = 15
    # S[1] = 0 (unchanged sinnce cluster size == 1)
    S, c1, c2 = pyamg.graph._split_improvement(m, d, dist_all, num_clusters=2)
    np.testing.assert_array_equal(S, [15, 0])
    np.testing.assert_array_equal(c1, [1, 6])
    np.testing.assert_array_equal(c2, [4, 6])
    print(c1, c2)

    # rebalance
    # new centers: 1, 4 (from above)
    newc, rebalance_change = \
    pyamg.graph._rebalance(
        G=G,
        c=c,
        m=m,
        d=d,
        dist_all=dist_all,
        num_clusters=2)
    np.testing.assert_array_equal(np.sort(newc), [1, 4])
    assert rebalance_change
