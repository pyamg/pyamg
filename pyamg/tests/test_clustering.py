"""Test clustering."""
import numpy as np
import pyamg.amg_core as amg_core
from pyamg.gallery import load_example
import scipy.sparse as sparse

from numpy.testing import TestCase, assert_equal, assert_array_equal


def canonical_graph(G):
    # convert to expected format
    # - remove diagonal entries
    # - all nonzero values = 1
    G = sparse.coo_matrix(G)

    mask = G.row != G.col
    G.row = G.row[mask]
    G.col = G.col[mask]
    G.data = G.data[mask]
    G.data[:] = 1
    return G


class TestClustering(TestCase):
    def setUp(self):
        cases = [None for i in range(5)]

        cluster_node_incidence_input = [None for i in range(5)]
        cluster_node_incidence_output = [None for i in range(5)]

        cluster_center_input = [None for i in range(5)]
        cluster_center_output = [None for i in range(5)]

        bellman_ford_input = [None for i in range(5)]
        bellman_ford_output = [None for i in range(5)]

        # bellman_ford_balanced_input = [None for i in range(5)]
        # bellman_ford_balanced_output = [None for i in range(5)]

        lloyd_cluster_input = [None for i in range(5)]
        lloyd_cluster_output = [None for i in range(5)]

        lloyd_cluster_exact_input = [None for i in range(5)]
        lloyd_cluster_exact_output = [None for i in range(5)]

        # (0) 6 node undirected, unit length
        # (1) 12 node undirected, unit length
        # (2) 16 node undirected, random length
        # (3) 16 node directed, random length
        # (4) 191 node unstructured finite element matrix

        # (0) 6 node undirected, unit length
        #
        # [3] ---- [4] ---- [5]
        #  |  \   / |  \   / |
        #  |  /   \ |  /   \ |
        # [0] ---- [1] ---- [2]
        xy = np.array([[0, 0],
                       [1, 0],
                       [2, 0],
                       [0, 1],
                       [1, 1],
                       [2, 1]])
        del xy
        G = np.zeros((6, 6))
        G[0, [1, 3, 4]] = 1
        G[1, [0, 2, 3, 4, 5]] = 1
        G[2, [1, 4, 5]] = 1
        G[3, [0, 1, 4]] = 1
        G[4, [0, 1, 2, 3, 5]] = 1
        G[5, [1, 2, 4]] = 1
        G[[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]] = [3, 5, 3, 3, 5, 3]
        G = sparse.csr_matrix(G)
        cases[0] = (G)

        cm = np.array([0, 1, 1, 0, 0, 1], dtype=np.int32)
        ICp = np.array([0, 3, 6], dtype=np.int32)
        ICi = np.array([0, 3, 4, 1, 2, 5], dtype=np.int32)
        L = np.array([0, 0, 1, 1, 2, 2], dtype=np.int32)
        cluster_node_incidence_input[0] = {'num_clusters': 2,
                                           'cm': cm}
        cluster_node_incidence_output[0] = {'ICp': ICp,
                                            'ICi': ICi,
                                            'L': L}

        cluster_center_input[0] = {'a': [0, 1],
                                   'num_clusters': 2,
                                   'cm': np.array([0, 1, 1, 0, 0, 1], dtype=np.int32),
                                   'ICp': np.array([0, 3, 6], dtype=np.int32),
                                   'ICi': np.array([0, 3, 4, 1, 2, 5], dtype=np.int32),
                                   'L': np.array([0, 0, 1, 1, 2, 2], dtype=np.int32)}
        cluster_center_output[0] = [0, 1]

        bellman_ford_input[0] = {'seeds': [0, 5]}
        bellman_ford_output[0] = {'cm': np.array([0, 0, 1, 0, 0, 1], dtype=np.int32),
                                  'd': np.array([0., 1., 1., 1., 1., 0.], dtype=G.dtype)}

        lloyd_cluster_input[0] = {'seeds': np.array([0, 5], dtype=np.int32)}
        lloyd_cluster_output[0] = {'cm': np.array([0, 0, 1, 0, 0, 1], dtype=np.int32),
                                   'd': np.array([1., 0., 0., 1., 0., 0.], dtype=G.dtype),
                                   'c': np.array([0, 5], dtype=np.int32)}

        lloyd_cluster_exact_input[0] = {'seeds': np.array([0, 5], dtype=np.int32)}
        lloyd_cluster_exact_output[0] = {'cm': np.array([0, 0, 1, 0, 1, 1], dtype=np.int32),
                                         'd': np.array([0, 1, 1, 1, 1, 0], dtype=G.dtype),
                                         'c': np.array([0, 2], dtype=np.int32)}

        # (1) 12 node undirected, unit length
        #
        #         _[1] ---- [7]
        #       /   |        |
        #     /     |        |
        # [0] ---- [2] ---- [6] ---- [8]
        #           |        |        |
        #           |        |        |
        #          [3] ---- [5] ---- [9] _
        #                    |        |    \
        #                    |        |      \
        #                   [4] ---- [10]---- [11]
        xy = np.array([[0, 2],
                       [1, 3],
                       [1, 2],
                       [1, 1],
                       [2, 0],
                       [2, 1],
                       [2, 2],
                       [2, 3],
                       [3, 2],
                       [3, 1],
                       [3, 0],
                       [4, 0]])
        del xy
        G = np.zeros((12, 12))
        G[0, [1, 2]] = 1
        G[1, [0, 2, 7]] = 1
        G[2, [0, 1, 3, 6]] = 1
        G[3, [2, 5]] = 1
        G[4, [5, 10]] = 1
        G[5, [3, 4, 6, 9]] = 1
        G[6, [2, 5, 7, 8]] = 1
        G[7, [1, 6]] = 1
        G[8, [6, 9]] = 1
        G[9, [5, 8, 10, 11]] = 1
        G[10, [4, 9, 11]] = 1
        G[11, [9, 10]] = 1
        G[np.arange(12), np.arange(12)] = [2, 3, 4, 2, 2, 4, 4, 2, 2, 4, 3, 2]
        G = sparse.csr_matrix(G)
        cases.append(G)

        cm = np.array([0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1], dtype=np.int32)
        ICp = np.array([0, 6, 12], dtype=np.int32)
        ICi = np.array([0, 1, 2, 3, 6, 7, 4, 5, 8, 9, 10, 11], dtype=np.int32)
        L = np.array([0, 1, 2, 3, 0, 1, 4, 5, 2, 3, 4, 5], dtype=np.int32)
        cluster_node_incidence_input.append({'num_clusters': 2,
                                             'cm': cm})
        cluster_node_incidence_output.append({'ICp': ICp,
                                              'ICi': ICi,
                                              'L': L})
        cluster_center_input[0] = {'a': [0, 1],
                                   'num_clusters': 2,
                                   'cm': np.array([0, 1, 1, 0, 0, 1], dtype=np.int32),
                                   'ICp': np.array([0, 3, 6], dtype=np.int32),
                                   'ICi': np.array([0, 3, 4, 1, 2, 5], dtype=np.int32),
                                   'L': np.array([0, 0, 1, 1, 2, 2], dtype=np.int32)}
        cluster_center_output[0] = [0, 1]
        # (2) 16 node undirected, random length (0,2)
        np.random.seed(2244369509)
        G.data[:] = np.random.rand(len(G.data)) * 2
        cases.append(G)

        # (3) 16 node directed, random length
        #         >[1] ---> [7]
        #       /   |        |
        #     /     v        v
        # [0] <--- [2] <--- [6] <--- [8]
        #           |        ^        ^
        #           v        |        |
        #          [3] ---> [5] <--- [9] <
        #                    |        ^    \
        #                    v        |      \
        #                   [4] ---> [10]---> [11]
        xy = np.array([[0, 2],
                       [1, 3],
                       [1, 2],
                       [1, 1],
                       [2, 0],
                       [2, 1],
                       [2, 2],
                       [2, 3],
                       [3, 2],
                       [3, 1],
                       [3, 0],
                       [4, 0]])
        del xy
        G = np.zeros((12, 12))
        G[0, [1]] = 1
        G[1, [2, 7]] = 1
        G[2, [0, 3]] = 1
        G[3, [5]] = 1
        G[4, [10]] = 1
        G[5, [4, 6]] = 1
        G[6, [2]] = 1
        G[7, [6]] = 1
        G[8, [6]] = 1
        G[9, [5, 8]] = 1
        G[10, [9, 11]] = 1
        G[11, [9]] = 1
        G = sparse.csr_matrix(G)
        np.random.seed(1664236979)
        G.data[:] = np.random.rand(len(G.data)) * 2
        cases.append(G)

        # (4) 191 node unstructured finite element matrix
        cases.append(load_example('unit_square')['A'])

        self.cases = cases
        self.cluster_node_incidence_input = cluster_node_incidence_input
        self.cluster_node_incidence_output = cluster_node_incidence_output
        self.cluster_center_input = cluster_center_input
        self.cluster_center_output = cluster_center_output
        self.bellman_ford_input = bellman_ford_input
        self.bellman_ford_output = bellman_ford_output
        # self.bellman_ford_balanced_input = bellman_ford_balanced_input
        # self.bellman_ford_balanced_output = bellman_ford_balanced_output
        self.lloyd_cluster_input = lloyd_cluster_input
        self.lloyd_cluster_output = lloyd_cluster_output
        self.lloyd_cluster_exact_input = lloyd_cluster_exact_input
        self.lloyd_cluster_exact_output = lloyd_cluster_exact_output

    def test_cluster_node_incidence(self):
        for A, argin, argout in zip(self.cases,
                                    self.cluster_node_incidence_input,
                                    self.cluster_node_incidence_output):

            if argin is not None:
                num_nodes = A.shape[0]
                num_clusters = argin['num_clusters']
                cm = argin['cm']
                ICp = -1*np.ones(num_clusters+1, dtype=np.int32)
                ICi = -1*np.ones(num_nodes, dtype=np.int32)
                L = -1*np.ones(num_nodes, dtype=np.int32)

                amg_core.cluster_node_incidence(num_nodes, num_clusters, cm, ICp, ICi, L)

                assert_array_equal(ICp, argout['ICp'])
                assert_array_equal(ICi, argout['ICi'])
                assert_array_equal(L, argout['L'])

    def test_cluster_center(self):
        for A, argin, argout in zip(self.cases,
                                    self.cluster_center_input,
                                    self.cluster_center_output):

            if argin is not None:
                for a, ccorrect in zip(argin['a'], argout):
                    num_nodes = A.shape[0]
                    num_clusters = argin['num_clusters']
                    cm = argin['cm']
                    ICp = argin['ICp']
                    ICi = argin['ICi']
                    L = argin['L']
                    c = amg_core.cluster_center(a,
                                                num_nodes, num_clusters,
                                                A.indptr, A.indices, A.data,
                                                cm,
                                                ICp, ICi, L)

                    assert_equal(c, ccorrect)

    def test_bellman_ford(self):
        for A, argin, argout in zip(self.cases,
                                    self.bellman_ford_input,
                                    self.bellman_ford_output):

            if argin is not None:
                seeds = argin['seeds']
                num_nodes = A.shape[0]

                mv = np.finfo(A.dtype).max
                d = mv * np.ones(num_nodes, dtype=A.dtype)
                d[seeds] = 0

                cm = -1 * np.ones(num_nodes, dtype=np.int32)
                cm[seeds] = np.arange(len(seeds))

                old_d = np.empty_like(d)

                iter = 0
                maxiter = 10
                while maxiter is None or iter < maxiter:
                    old_d[:] = d

                    amg_core.bellman_ford(num_nodes,
                                          A.indptr, A.indices, A.data,
                                          d, cm)

                    if (old_d == d).all():
                        break

                    iter += 1

                assert_array_equal(d, argout['d'])
                assert_array_equal(cm, argout['cm'])

    # def test_bellman_ford_balanced(self):
    #     for A, argin, argout in zip(self.cases,
    #                                 self.bellman_ford_balanced_input,
    #                                 self.bellman_ford_balanced_output):

    #         if argin is not None:
    #             seeds = argin['seeds']
    #             num_nodes = A.shape[0]
    #             num_clusters = len(seeds)

    #             mv = np.finfo(A.dtype).max
    #             d = mv * np.ones(num_nodes, dtype=A.dtype)
    #             d[seeds] = 0

    #             cm = 0 * np.ones(num_nodes, dtype=np.int32)
    #             cm[seeds] = np.arange(len(seeds))

    #             amg_core.bellman_ford_balanced(num_nodes, num_clusters,
    #                                            A.indptr, A.indices, A.data,
    #                                            d, cm)

    #             assert_array_equal(d, argout['d'])
    #             assert_array_equal(cm, argout['cm'])

    def test_lloyd_cluster(self):
        for A, argin, argout in zip(self.cases,
                                    self.lloyd_cluster_input,
                                    self.lloyd_cluster_output):

            if argin is not None:
                seeds = argin['seeds']
                num_nodes = A.shape[0]
                c = seeds.copy()
                num_clusters = len(seeds)

                mv = np.finfo(A.dtype).max
                d = mv * np.ones(num_nodes, dtype=A.dtype)
                d[seeds] = 0

                cm = -1 * np.ones(num_nodes, dtype=np.int32)
                cm[seeds] = seeds

                amg_core.lloyd_cluster(num_nodes,
                                       A.indptr, A.indices, A.data,
                                       num_clusters,
                                       d, cm, c)

                assert_array_equal(d, argout['d'])
                assert_array_equal(cm, argout['cm'])
                assert_array_equal(c, argout['c'])

    def test_lloyd_cluster_exact(self):
        for A, argin, argout in zip(self.cases,
                                    self.lloyd_cluster_exact_input,
                                    self.lloyd_cluster_exact_output):

            if argin is not None:
                seeds = argin['seeds']
                num_nodes = A.shape[0]
                c = seeds.copy()
                num_clusters = len(seeds)

                mv = np.finfo(A.dtype).max
                d = mv * np.ones(num_nodes, dtype=A.dtype)
                d[seeds] = 0

                cm = -1 * np.ones(num_nodes, dtype=np.int32)
                cm[seeds] = seeds

                amg_core.lloyd_cluster_exact(num_nodes,
                                             A.indptr, A.indices, A.data,
                                             num_clusters,
                                             d, cm, c)

                assert_array_equal(d, argout['d'])
                assert_array_equal(cm, argout['cm'])
                assert_array_equal(c, argout['c'])
