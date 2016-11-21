import numpy
from numpy import ones, eye, zeros, bincount, empty, asarray, array
from numpy.random import seed
from scipy import rand
from scipy.sparse import csr_matrix, coo_matrix

from pyamg.gallery import poisson, load_example
from pyamg.graph import maximal_independent_set, vertex_coloring,\
    bellman_ford, lloyd_cluster, connected_components, max_value
from pyamg import amg_core

from numpy.testing import TestCase, assert_equal


def canonical_graph(G):
    # convert to expected format
    # - remove diagonal entries
    # - all nonzero values = 1
    G = coo_matrix(G)

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
        assert((mis[G.row] + mis[G.col]).max() <= 1)
    # all non-set vertices have set neighbor
    assert((mis + G*mis).min() == 1)


def assert_is_vertex_coloring(G, c):
    G = canonical_graph(G)

    # no colors joined by an edge
    assert((c[G.row] != c[G.col]).all())
    # all colors up to K occur at least once
    assert((bincount(c) > 0).all())


class TestGraph(TestCase):
    def setUp(self):
        cases = []
        seed(0)

        for i in range(5):
            A = rand(8, 8) > 0.5
            cases.append(canonical_graph(A + A.T).astype(float))

        cases.append(zeros((1, 1)))
        cases.append(zeros((2, 2)))
        cases.append(zeros((8, 8)))
        cases.append(ones((2, 2)) - eye(2))
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
        assert_equal(maximal_independent_set(eye(2)), [1, 1])

        for algo in ['serial', 'parallel']:
            for G in self.cases:
                mis = maximal_independent_set(G, algo=algo)
                assert_is_mis(G, mis)

        for G in self.cases:
            for k in [1, 2, 3, 4]:
                mis = maximal_independent_set(G, k=k)
                if k > 1:
                    G = (G + eye(G.shape[0]))**k
                    G = canonical_graph(G)
                assert_is_mis(G, mis)

    def test_vertex_coloring(self):
        # test that method works with diagonal entries
        assert_equal(vertex_coloring(eye(1)), [0])
        assert_equal(vertex_coloring(eye(3)), [0, 0, 0])
        assert_equal(sorted(vertex_coloring(ones((3, 3)))), [0, 1, 2])

        for method in ['MIS', 'JP', 'LDF']:
            for G in self.cases:
                c = vertex_coloring(G, method=method)
                assert_is_vertex_coloring(G, c)

    def test_bellman_ford(self):
        numpy.random.seed(0)

        for G in self.cases:
            G.data = rand(G.nnz)
            N = G.shape[0]

            for n_seeds in [int(N/20), int(N/10), N-2, N]:
                if n_seeds > G.shape[0] or n_seeds < 1:
                    continue

                seeds = numpy.random.permutation(N)[:n_seeds]
                D_expected, S_expected = reference_bellman_ford(G, seeds)
                D_result, S_result = bellman_ford(G, seeds)

                assert_equal(D_result, D_expected)
                assert_equal(S_result, S_expected)

    def test_lloyd_cluster(self):
        numpy.random.seed(0)

        for G in self.cases:
            G.data = rand(G.nnz)

            for n_seeds in [5]:
                if n_seeds > G.shape[0]:
                    continue

                distances, clusters, centers = lloyd_cluster(G, n_seeds)


class TestComplexGraph(TestCase):
    def setUp(self):
        cases = []
        seed(0)

        for i in range(5):
            A = rand(8, 8) > 0.5
            cases.append(canonical_graph(A + A.T).astype(float))

        cases = [canonical_graph(G)+1.0j*canonical_graph(G) for G in cases]

        self.cases = cases

    def test_maximal_independent_set(self):
        # test that method works with diagonal entries
        assert_equal(maximal_independent_set(eye(2)), [1, 1])

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
        numpy.random.seed(0)

        for G in self.cases:
            G.data = rand(G.nnz) + 1.0j*rand(G.nnz)

            for n_seeds in [5]:
                if n_seeds > G.shape[0]:
                    continue

                distances, clusters, centers = lloyd_cluster(G, n_seeds)


class TestVertexColorings(TestCase):
    def setUp(self):
        #      3---4
        #    / | / |
        #  0---1---2
        G0 = array([[0, 1, 0, 1, 0],
                    [1, 0, 1, 1, 1],
                    [0, 1, 0, 0, 1],
                    [1, 1, 0, 0, 1],
                    [0, 1, 1, 1, 0]])
        self.G0 = csr_matrix(G0)
        # make sure graph is symmetric
        assert_equal((self.G0 - self.G0.T).nnz, 0)

        #  2        5
        #  | \    / |
        #  0--1--3--4
        G1 = array([[0, 1, 1, 0, 0, 0],
                    [1, 0, 1, 1, 0, 0],
                    [1, 1, 0, 0, 0, 0],
                    [0, 1, 0, 0, 1, 1],
                    [0, 0, 0, 1, 0, 1],
                    [0, 0, 0, 1, 1, 0]])
        self.G1 = csr_matrix(G1)
        # make sure graph is symmetric
        assert_equal((self.G1 - self.G1.T).nnz, 0)

    def test_vertex_coloring_JP(self):
        fn = amg_core.vertex_coloring_jones_plassmann

        weights = array([0.8, 0.1, 0.9, 0.7, 0.6], dtype='float64')
        coloring = empty(5, dtype='intc')
        fn(self.G0.shape[0], self.G0.indptr, self.G0.indices, coloring,
           weights)
        assert_equal(coloring, [2, 0, 1, 1, 2])

        weights = array([0.1, 0.2, 0.3, 0.1, 0.2, 0.3], dtype='float64')
        coloring = empty(6, dtype='intc')
        fn(self.G1.shape[0], self.G1.indptr, self.G1.indices, coloring,
           weights)
        assert_equal(coloring, [2, 0, 1, 1, 2, 0])

    def test_vertex_coloring_LDF(self):
        fn = amg_core.vertex_coloring_LDF

        weights = array([0.8, 0.1, 0.9, 0.7, 0.6], dtype='float64')
        coloring = empty(5, dtype='intc')
        fn(self.G0.shape[0], self.G0.indptr, self.G0.indices, coloring,
           weights)
        assert_equal(coloring, [2, 0, 1, 1, 2])

        weights = array([0.1, 0.2, 0.3, 0.1, 0.2, 0.3], dtype='float64')
        coloring = empty(6, dtype='intc')
        fn(self.G1.shape[0], self.G1.indptr, self.G1.indices, coloring,
           weights)
        assert_equal(coloring, [2, 0, 1, 2, 1, 0])


def test_breadth_first_search():
    from pyamg.graph import breadth_first_search

    BFS = breadth_first_search

    G = csr_matrix([[0, 1, 0, 0],
                    [1, 0, 1, 0],
                    [0, 1, 0, 1],
                    [0, 0, 1, 0]])

    assert_equal(BFS(G, 0)[1], [0, 1, 2, 3])
    assert_equal(BFS(G, 1)[1], [1, 0, 1, 2])
    assert_equal(BFS(G, 2)[1], [2, 1, 0, 1])
    assert_equal(BFS(G, 3)[1], [3, 2, 1, 0])

    G = csr_matrix([[0, 1, 0, 0],
                    [1, 0, 1, 0],
                    [0, 1, 0, 0],
                    [0, 0, 0, 0]])

    assert_equal(BFS(G, 0)[1], [0, 1, 2, -1])
    assert_equal(BFS(G, 1)[1], [1, 0, 1, -1])
    assert_equal(BFS(G, 2)[1], [2, 1, 0, -1])
    assert_equal(BFS(G, 3)[1], [-1, -1, -1, 0])


def test_connected_components():

    cases = []
    cases.append(csr_matrix([[0, 1, 0, 0],
                             [1, 0, 1, 0],
                             [0, 1, 0, 1],
                             [0, 0, 1, 0]]))

    cases.append(csr_matrix([[0, 1, 0, 0],
                             [1, 0, 0, 0],
                             [0, 0, 0, 1],
                             [0, 0, 1, 0]]))

    cases.append(csr_matrix([[0, 1, 0, 0],
                             [1, 0, 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 0, 0]]))

    cases.append(csr_matrix([[0, 0, 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 0, 0]]))

    #  2        5
    #  | \    / |
    #  0--1--3--4
    cases.append(csr_matrix([[0, 1, 1, 0, 0, 0],
                             [1, 0, 1, 1, 0, 0],
                             [1, 1, 0, 0, 0, 0],
                             [0, 1, 0, 0, 1, 1],
                             [0, 0, 0, 1, 0, 1],
                             [0, 0, 0, 1, 1, 0]]))

    #  2        5
    #  | \    / |
    #  0  1--3--4
    cases.append(csr_matrix([[0, 0, 1, 0, 0, 0],
                             [0, 0, 1, 1, 0, 0],
                             [1, 1, 0, 0, 0, 0],
                             [0, 1, 0, 0, 1, 1],
                             [0, 0, 0, 1, 0, 1],
                             [0, 0, 0, 1, 1, 0]]))

    #  2        5
    #  | \    / |
    #  0--1  3--4
    cases.append(csr_matrix([[0, 1, 1, 0, 0, 0],
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
            """convert array to set of sets format"""
            D = {}
            for i in set(arr):
                D[i] = set()
            for n, i in enumerate(arr):
                D[i].add(n)
            return set([frozenset(s) for s in D.values()])

        result = array_to_set_of_sets(result)
        expected = reference_connected_components(G)

        assert_equal(result, expected)


def test_complex_connected_components():

    cases = []
    cases.append(csr_matrix([[0, 1, 0, 0],
                             [1, 0, 1, 0],
                             [0, 1, 0, 1],
                             [0, 0, 1, 0]]))

    cases.append(csr_matrix([[0, 1, 0, 0],
                             [1, 0, 0, 0],
                             [0, 0, 0, 1],
                             [0, 0, 1, 0]]))

    cases.append(csr_matrix([[0, 1, 0, 0],
                             [1, 0, 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 0, 0]]))

    cases.append(csr_matrix([[0, 0, 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 0, 0]]))

    #  2        5
    #  | \    / |
    #  0--1--3--4
    cases.append(csr_matrix([[0, 1, 1, 0, 0, 0],
                             [1, 0, 1, 1, 0, 0],
                             [1, 1, 0, 0, 0, 0],
                             [0, 1, 0, 0, 1, 1],
                             [0, 0, 0, 1, 0, 1],
                             [0, 0, 0, 1, 1, 0]]))

    #  2        5
    #  | \    / |
    #  0  1--3--4
    cases.append(csr_matrix([[0, 0, 1, 0, 0, 0],
                             [0, 0, 1, 1, 0, 0],
                             [1, 1, 0, 0, 0, 0],
                             [0, 1, 0, 0, 1, 1],
                             [0, 0, 0, 1, 0, 1],
                             [0, 0, 0, 1, 1, 0]]))

    #  2        5
    #  | \    / |
    #  0--1  3--4
    cases.append(csr_matrix([[0, 1, 1, 0, 0, 0],
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
            """convert array to set of sets format"""
            D = {}
            for i in set(arr):
                D[i] = set()
            for n, i in enumerate(arr):
                D[i].add(n)
            return set([frozenset(s) for s in D.values()])

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


def reference_bellman_ford(G, seeds):
    G = G.tocoo()
    N = G.shape[0]

    seeds = asarray(seeds, dtype='intc')

    distances = empty(N, dtype=G.dtype)
    distances[:] = max_value(G.dtype)
    distances[seeds] = 0

    nearest_seed = empty(N, dtype='intc')
    nearest_seed[:] = -1
    nearest_seed[seeds] = seeds

    while True:
        update = False

        for (i, j, v) in zip(G.row, G.col, G.data):

            if distances[j] + v < distances[i]:
                update = True
                distances[i] = distances[j] + v
                nearest_seed[i] = nearest_seed[j]

        if not update:
            break

    return (distances, nearest_seed)
