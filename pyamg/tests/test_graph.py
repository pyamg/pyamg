"""Test graph routings."""
import numpy as np
from scipy import sparse

import pytest

from numpy.testing import TestCase, assert_equal

from pyamg.gallery import poisson, load_example
from pyamg.graph import (maximal_independent_set, vertex_coloring,
                         bellman_ford, lloyd_cluster, connected_components,
                         metis_partition)
from pyamg.graph_ref import bellman_ford_reference, bellman_ford_balanced_reference
from pyamg import amg_core


def canonical_graph(G):
    # convert to expected format
    # - remove diagonal entries
    # - all nonzero values = 1
    G = sparse.coo_array(G)

    mask = G.row != G.col
    G.row = G.row[mask]
    G.col = G.col[mask]
    G.data = G.data[mask]
    G.data[:] = 1
    return G


def assert_is_mis(G, mis):
    G = canonical_graph(G)

    # no MIS vertices joined by an edge
    if G.nnz > 0:
        assert (mis[G.row] + mis[G.col]).max() <= 1
    # all non-set vertices have set neighbor
    assert (mis + G@mis).min() == 1


def assert_is_vertex_coloring(G, c):
    G = canonical_graph(G)

    # no colors joined by an edge
    assert (c[G.row] != c[G.col]).all()
    # all colors up to K occur at least once
    assert (np.bincount(c) > 0).all()


class TestGraph(TestCase):
    def setUp(self):
        cases = []
        np.random.seed(651978631)

        for _i in range(5):
            A = np.random.rand(8, 8) > 0.5
            cases.append(canonical_graph(A + A.T).astype(float))

        cases.append(np.zeros((1, 1)))
        cases.append(np.zeros((2, 2)))
        cases.append(np.zeros((8, 8)))
        cases.append(np.ones((2, 2)) - np.eye(2))
        cases.append(poisson((5,)))
        cases.append(poisson((5, 5)))
        cases.append(poisson((11, 11)))
        cases.append(poisson((5, 5, 5)))
        for name in ['airfoil', 'bar', 'knot']:
            cases.append(load_example(name)['A'])

        cases = [canonical_graph(G) for G in cases]

        self.cases = cases

    def test_maximal_independent_set(self):
        # test that method works with diagonal entries
        assert_equal(maximal_independent_set(np.eye(2)), [1, 1])

        for algo in ['serial', 'parallel']:
            for G in self.cases:
                mis = maximal_independent_set(G, algo=algo)
                assert_is_mis(G, mis)

        for G in self.cases:
            for k in [1, 2, 3, 4]:
                mis = maximal_independent_set(G, k=k)
                if k > 1:
                    G = sparse.linalg.matrix_power(G + np.eye(G.shape[0]), k)
                    G = canonical_graph(G)
                assert_is_mis(G, mis)

    def test_vertex_coloring(self):
        # test that method works with diagonal entries
        assert_equal(vertex_coloring(np.eye(1)), [0])
        assert_equal(vertex_coloring(np.eye(3)), [0, 0, 0])
        assert_equal(sorted(vertex_coloring(np.ones((3, 3)))), [0, 1, 2])

        for method in ['MIS', 'JP', 'LDF']:
            for G in self.cases:
                c = vertex_coloring(G, method=method)
                assert_is_vertex_coloring(G, c)

    def test_bellman_ford(self):
        """Test pile of cases against reference implementation."""
        np.random.seed(1643502758)

        for G in self.cases:
            G.data = np.random.rand(G.nnz)
            N = G.shape[0]

            for n_clusters in [int(N/20), int(N/10), N-2, N]:
                if n_clusters > G.shape[0] or n_clusters < 1:
                    continue

                centers = np.random.permutation(N)[:n_clusters]

                for method in ['standard', 'balanced']:
                    d_result, m_result, p_result = bellman_ford(G, centers, method=method)
                    if method == 'standard':
                        d_expected, m_expected, p_expected = \
                            bellman_ford_reference(G, centers)
                    if method == 'balanced':
                        d_expected, m_expected, p_expected = \
                            bellman_ford_balanced_reference(G, centers)

                assert_equal(d_result, d_expected)
                assert_equal(m_result, m_expected)
                assert_equal(p_result, p_expected)

    def test_bellman_ford_reference(self):
        Edges = np.array([[1, 4],
                          [3, 1],
                          [1, 3],
                          [0, 1],
                          [0, 2],
                          [3, 2],
                          [1, 2],
                          [4, 3]], dtype=np.int32)
        w = np.array([2, 1, 2, 1, 4, 5, 3, 1], dtype=float)
        G = sparse.coo_array((w, (Edges[:, 0], Edges[:, 1])))
        distances_FROM_center = np.array([[0.,     1.,     4., 3.,     3.],
                                          [np.inf, 0.,     3., 2.,     2.],
                                          [np.inf, np.inf, 0., np.inf, np.inf],
                                          [np.inf, 1.,     4., 0.,     3.],
                                          [np.inf, 2.,     5., 1.,     0.]])

        for center in range(5):
            distance, _nearest, _predecessor = bellman_ford_reference(G, [center])
            assert_equal(distance, distances_FROM_center[center])

            distance, _nearest, _predecessor = bellman_ford(G, [center])
            assert_equal(distance, distances_FROM_center[center])

    def test_lloyd_cluster(self):
        np.random.seed(3125088753)

        for G in self.cases:
            G.data = np.random.rand(G.nnz)

            for n_clusters in [5]:
                if n_clusters > G.shape[0]:
                    continue

                _clusters, _centers = lloyd_cluster(G, n_clusters)


class TestComplexGraph(TestCase):
    def setUp(self):
        cases = []
        np.random.seed(3084315563)

        for _i in range(5):
            A = np.random.rand(8, 8) > 0.5
            cases.append(canonical_graph(A + A.T).astype(float))

        cases = [canonical_graph(G)+1.0j*canonical_graph(G) for G in cases]

        self.cases = cases

    def test_maximal_independent_set(self):
        # test that method works with diagonal entries
        assert_equal(maximal_independent_set(np.eye(2)), [1, 1])

        for algo in ['serial', 'parallel']:
            for G in self.cases:
                mis = maximal_independent_set(G, algo=algo)
                assert_is_mis(G, mis)

    def test_vertex_coloring(self):
        for method in ['MIS', 'JP', 'LDF']:
            for G in self.cases:
                c = vertex_coloring(G, method=method)
                assert_is_vertex_coloring(G, c)

    def test_lloyd_cluster(self):
        np.random.seed(2099568097)

        for G in self.cases:
            G.data = np.random.rand(G.nnz) + 1.0j*np.random.rand(G.nnz)

            for n_clusters in [5]:
                if n_clusters > G.shape[0]:
                    continue

                _clusters, _centers = lloyd_cluster(G, n_clusters)


class TestVertexColorings(TestCase):
    def setUp(self):
        #      3---4
        #    / | / |
        #  0---1---2
        G0 = np.array([[0, 1, 0, 1, 0],
                       [1, 0, 1, 1, 1],
                       [0, 1, 0, 0, 1],
                       [1, 1, 0, 0, 1],
                       [0, 1, 1, 1, 0]])
        self.G0 = sparse.csr_array(G0)
        # make sure graph is symmetric
        assert_equal((self.G0 - self.G0.T).nnz, 0)

        #  2        5
        #  | \    / |
        #  0--1--3--4
        G1 = np.array([[0, 1, 1, 0, 0, 0],
                       [1, 0, 1, 1, 0, 0],
                       [1, 1, 0, 0, 0, 0],
                       [0, 1, 0, 0, 1, 1],
                       [0, 0, 0, 1, 0, 1],
                       [0, 0, 0, 1, 1, 0]])
        self.G1 = sparse.csr_array(G1)
        # make sure graph is symmetric
        assert_equal((self.G1 - self.G1.T).nnz, 0)

    def test_vertex_coloring_JP(self):
        fn = amg_core.vertex_coloring_jones_plassmann

        weights = np.array([0.8, 0.1, 0.9, 0.7, 0.6], dtype='float64')
        coloring = np.empty(5, dtype='intc')
        fn(self.G0.shape[0], self.G0.indptr, self.G0.indices, coloring,
           weights)
        assert_equal(coloring, [2, 0, 1, 1, 2])

        weights = np.array([0.1, 0.2, 0.3, 0.1, 0.2, 0.3], dtype='float64')
        coloring = np.empty(6, dtype='intc')
        fn(self.G1.shape[0], self.G1.indptr, self.G1.indices, coloring,
           weights)
        assert_equal(coloring, [2, 0, 1, 1, 2, 0])

    def test_vertex_coloring_LDF(self):
        fn = amg_core.vertex_coloring_LDF

        weights = np.array([0.8, 0.1, 0.9, 0.7, 0.6], dtype='float64')
        coloring = np.empty(5, dtype='intc')
        fn(self.G0.shape[0], self.G0.indptr, self.G0.indices, coloring,
           weights)
        assert_equal(coloring, [2, 0, 1, 1, 2])

        weights = np.array([0.1, 0.2, 0.3, 0.1, 0.2, 0.3], dtype='float64')
        coloring = np.empty(6, dtype='intc')
        fn(self.G1.shape[0], self.G1.indptr, self.G1.indices, coloring,
           weights)
        assert_equal(coloring, [2, 0, 1, 2, 1, 0])


def test_breadth_first_search():
    from pyamg.graph import breadth_first_search

    BFS = breadth_first_search

    G = sparse.csr_array([[0, 1, 0, 0],
                          [1, 0, 1, 0],
                          [0, 1, 0, 1],
                          [0, 0, 1, 0]])

    assert_equal(BFS(G, 0)[1], [0, 1, 2, 3])
    assert_equal(BFS(G, 1)[1], [1, 0, 1, 2])
    assert_equal(BFS(G, 2)[1], [2, 1, 0, 1])
    assert_equal(BFS(G, 3)[1], [3, 2, 1, 0])

    G = sparse.csr_array([[0, 1, 0, 0],
                          [1, 0, 1, 0],
                          [0, 1, 0, 0],
                          [0, 0, 0, 0]])

    assert_equal(BFS(G, 0)[1], [0, 1, 2, -1])
    assert_equal(BFS(G, 1)[1], [1, 0, 1, -1])
    assert_equal(BFS(G, 2)[1], [2, 1, 0, -1])
    assert_equal(BFS(G, 3)[1], [-1, -1, -1, 0])


def test_connected_components():

    cases = []
    cases.append(sparse.csr_array([[0, 1, 0, 0],
                                   [1, 0, 1, 0],
                                   [0, 1, 0, 1],
                                   [0, 0, 1, 0]]))

    cases.append(sparse.csr_array([[0, 1, 0, 0],
                                   [1, 0, 0, 0],
                                   [0, 0, 0, 1],
                                   [0, 0, 1, 0]]))

    cases.append(sparse.csr_array([[0, 1, 0, 0],
                                   [1, 0, 0, 0],
                                   [0, 0, 0, 0],
                                   [0, 0, 0, 0]]))

    cases.append(sparse.csr_array([[0, 0, 0, 0],
                                   [0, 0, 0, 0],
                                   [0, 0, 0, 0],
                                   [0, 0, 0, 0]]))

    #  2        5
    #  | \    / |
    #  0--1--3--4
    cases.append(sparse.csr_array([[0, 1, 1, 0, 0, 0],
                                   [1, 0, 1, 1, 0, 0],
                                   [1, 1, 0, 0, 0, 0],
                                   [0, 1, 0, 0, 1, 1],
                                   [0, 0, 0, 1, 0, 1],
                                   [0, 0, 0, 1, 1, 0]]))

    #  2        5
    #  | \    / |
    #  0  1--3--4
    cases.append(sparse.csr_array([[0, 0, 1, 0, 0, 0],
                                   [0, 0, 1, 1, 0, 0],
                                   [1, 1, 0, 0, 0, 0],
                                   [0, 1, 0, 0, 1, 1],
                                   [0, 0, 0, 1, 0, 1],
                                   [0, 0, 0, 1, 1, 0]]))

    #  2        5
    #  | \    / |
    #  0--1  3--4
    cases.append(sparse.csr_array([[0, 1, 1, 0, 0, 0],
                                   [1, 0, 1, 0, 0, 0],
                                   [1, 1, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 1, 1],
                                   [0, 0, 0, 1, 0, 1],
                                   [0, 0, 0, 1, 1, 0]]))

    # Compare to reference implementation #
    for G in cases:
        result = connected_components(G)

        assert_equal(result.min(), 0)

        def array_to_set_of_sets(arr):
            """Convert array to set of sets format."""
            D = {}
            for i in set(arr):
                D[i] = set()
            for n, i in enumerate(arr):
                D[i].add(n)
            return {frozenset(s) for s in D.values()}

        result = array_to_set_of_sets(result)
        expected = reference_connected_components(G)

        assert_equal(result, expected)


def test_complex_connected_components():

    cases = []
    cases.append(sparse.csr_array([[0, 1, 0, 0],
                                   [1, 0, 1, 0],
                                   [0, 1, 0, 1],
                                   [0, 0, 1, 0]]))

    cases.append(sparse.csr_array([[0, 1, 0, 0],
                                   [1, 0, 0, 0],
                                   [0, 0, 0, 1],
                                   [0, 0, 1, 0]]))

    cases.append(sparse.csr_array([[0, 1, 0, 0],
                                   [1, 0, 0, 0],
                                   [0, 0, 0, 0],
                                   [0, 0, 0, 0]]))

    cases.append(sparse.csr_array([[0, 0, 0, 0],
                                   [0, 0, 0, 0],
                                   [0, 0, 0, 0],
                                   [0, 0, 0, 0]]))

    #  2        5
    #  | \    / |
    #  0--1--3--4
    cases.append(sparse.csr_array([[0, 1, 1, 0, 0, 0],
                                   [1, 0, 1, 1, 0, 0],
                                   [1, 1, 0, 0, 0, 0],
                                   [0, 1, 0, 0, 1, 1],
                                   [0, 0, 0, 1, 0, 1],
                                   [0, 0, 0, 1, 1, 0]]))

    #  2        5
    #  | \    / |
    #  0  1--3--4
    cases.append(sparse.csr_array([[0, 0, 1, 0, 0, 0],
                                   [0, 0, 1, 1, 0, 0],
                                   [1, 1, 0, 0, 0, 0],
                                   [0, 1, 0, 0, 1, 1],
                                   [0, 0, 0, 1, 0, 1],
                                   [0, 0, 0, 1, 1, 0]]))

    #  2        5
    #  | \    / |
    #  0--1  3--4
    cases.append(sparse.csr_array([[0, 1, 1, 0, 0, 0],
                                   [1, 0, 1, 0, 0, 0],
                                   [1, 1, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 1, 1],
                                   [0, 0, 0, 1, 0, 1],
                                   [0, 0, 0, 1, 1, 0]]))

    # Create complex data entries
    cases = [G+1.0j*G for G in cases]

    # Compare to reference implementation #
    for G in cases:
        result = connected_components(G)

        assert_equal(result.min(), 0)

        def array_to_set_of_sets(arr):
            """Convert array to set of sets format."""
            D = {}
            for i in set(arr):
                D[i] = set()
            for n, i in enumerate(arr):
                D[i].add(n)
            return {frozenset(s) for s in D.values()}

        result = array_to_set_of_sets(result)
        expected = reference_connected_components(G)

        assert_equal(result, expected)


# reference implementations #
def reference_connected_components(G):
    G = G.tocsr()
    N = G.shape[0]

    def DFS(i, G, component, visited):
        if i not in visited:
            component.add(i)
            visited.add(i)
            for j in G.indices[G.indptr[i]:G.indptr[i+1]]:
                DFS(j, G, component, visited)

    visited = set()
    components = set()
    for i in range(N):
        if i not in visited:
            component = set()
            DFS(i, G, component, visited)
            components.add(frozenset(component))

    return components


def test_metis():
    # check if pymetis is available before testing
    metis = pytest.importorskip('pymetis')  # noqa: F841

    # 0        4
    # | \    / |
    # 1--2--3--5
    # target [0, 0, 0, 1, 1, 1] or [1, 1, 1, 0, 0, 0]
    Edges = np.array([[0, 1],
                      [0, 2],
                      [1, 2],
                      [2, 3],
                      [3, 4],
                      [3, 5],
                      [4, 5]])
    w = np.ones(Edges.shape[0], dtype=int)
    G = sparse.coo_array((w, (Edges[:, 0], Edges[:, 1])), shape=(6, 6))
    G = G + G.T  # undirected
    G = G.tocoo()

    # set fake diagonal (which gets zeroed)
    G.setdiag(4.4)
    G = G.tocsr()

    parts = metis_partition(G, nparts=2)

    # check left part
    assert_equal(parts[:3], parts[0]*np.ones(3, dtype=int))

    # check right part
    assert_equal(parts[3:], parts[3]*np.ones(3, dtype=int))

    # 0 -- 1 -- 2
    # |    +    +
    # 3 ++ 4 -- 5
    # |    |    |
    # 6 ++ 7 -- 8
    # make edges ++ small ~1
    # make edges -- large ~10
    # target [0, 0, 0, 0, 1, 1, 0, 1, 1]
    #     or [1, 1, 1, 1, 0, 0, 1, 0, 0]
    Edges = np.array([[0, 1], [0, 3],
                      [1, 0], [1, 4], [1, 2],
                      [2, 1], [2, 5],
                      [3, 0], [3, 4], [3, 6],
                      [4, 1], [4, 3], [4, 5], [4, 7],
                      [5, 2], [5, 4], [5, 8],
                      [6, 3], [6, 7],
                      [7, 4], [7, 6], [7, 8],
                      [8, 5], [8, 7]])
    w = 10*np.ones(Edges.shape[0], dtype=int)
    for i, e in enumerate(Edges):
        if np.array_equal(np.sort([1, 4]), np.sort(e)) or \
           np.array_equal(np.sort([2, 5]), np.sort(e)) or \
           np.array_equal(np.sort([3, 4]), np.sort(e)) or \
           np.array_equal(np.sort([6, 7]), np.sort(e)):
            w[i] = 1
    G = sparse.coo_array((w, (Edges[:, 0], Edges[:, 1])), shape=(9, 9))
    G = G.tocsr()

    parts = metis_partition(G, nparts=2)

    # check first part
    I = [0, 1, 2, 3, 6]
    assert_equal(parts[I], parts[I[0]]*np.ones(5, dtype=int))

    # check second part
    I = [4, 5, 7, 8]
    assert_equal(parts[I], parts[I[0]]*np.ones(4, dtype=int))
