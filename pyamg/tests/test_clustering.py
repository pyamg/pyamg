"""Test clustering routines in amg_core."""
import numpy as np
import pyamg.amg_core as amg_core
from pyamg.gallery import load_example
from pyamg.graph import bellman_ford_reference, bellman_ford_balanced_reference
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
        cases_bellman_ford = []
        cases_bellman_ford_balanced = []

        # (0) 6 node undirected, unit length
        # (1) 12 node undirected, unit length
        # (2) 16 node undirected, random length
        # (3) 16 node directed, random length
        # (4) 191 node unstructured finite element matrix
        # (5) 5 nodes in a line
        # (6) 5 nodes in a graph

        ############################################################
        # (0) 6 node undirected, unit length
        ############################################################
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

        case = {}
        case['id'] = 0
        case['G'] = G
        case['input'] = {'centers': np.array([0, 5], dtype=np.int32)}
        case['output'] = {'m': np.array([0, 0, 1, 0, 0, 1], dtype=np.int32),
                          'd': np.array([0., 1., 1., 1., 1., 0.], dtype=G.dtype)}
        cases_bellman_ford.append(case)

        case = dict(case)
        del case['output']
        cases_bellman_ford_balanced.append(case)

        #cluster_node_incidence_input[0] = {'num_clusters': 2,
        #                                   'cm': np.array([0, 1, 1, 0, 0, 1], dtype=np.int32)}
        #cluster_node_incidence_output[0] = {'ICp': np.array([0, 3, 6], dtype=np.int32),
        #                                    'ICi': np.array([0, 3, 4, 1, 2, 5], dtype=np.int32),
        #                                    'L': np.array([0, 0, 1, 1, 2, 2], dtype=np.int32)}

        #cluster_center_input[0] = {'a': [0, 1],
        #                           'num_clusters': 2,
        #                           'cm': np.array([0, 1, 1, 0, 0, 1], dtype=np.int32),
        #                           'ICp': np.array([0, 3, 6], dtype=np.int32),
        #                           'ICi': np.array([0, 3, 4, 1, 2, 5], dtype=np.int32),
        #                           'L': np.array([0, 0, 1, 1, 2, 2], dtype=np.int32)}
        #cluster_center_output[0] = [0, 1]

        #bellman_ford_balanced_input[0] = {'centers': np.array([0, 5], dtype=np.int32)}
        #bellman_ford_balanced_output[0] = {'cm': np.array([0, 1, 1, 0, 0, 1], dtype=np.int32),
        #                                   'd': np.array([0., 1., 1., 1., 1., 0.], dtype=G.dtype)}

        #lloyd_cluster_input[0] = {'centers': np.array([0, 5], dtype=np.int32)}
        #lloyd_cluster_output[0] = {'m': np.array([0, 0, 1, 0, 0, 1], dtype=np.int32),
        #                           'd': np.array([1., 0., 0., 1., 0., 0.], dtype=G.dtype),
        #                           'c': np.array([0, 5], dtype=np.int32)}

        #lloyd_cluster_exact_input[0] = {'centers': np.array([0, 5], dtype=np.int32)}
        #lloyd_cluster_exact_output[0] = {'cm': np.array([0, 0, 1, 0, 1, 1], dtype=np.int32),
        #                                 'd': np.array([0., 1., 1., 1., 1., 0.], dtype=G.dtype),
        #                                 'c': np.array([0, 2], dtype=np.int32)}

        ############################################################
        # (1) 12 node undirected, unit length
        ############################################################
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

        case = {}
        case['id'] = 1
        case['G'] = G
        case['input'] = {'centers': np.array([0, 1], dtype=np.int32)}
        cases_bellman_ford.append(case)
        cases_bellman_ford_balanced.append(case)

        #cluster_node_incidence_input.append({'num_clusters': 2,
        #                                     'cm': np.array([0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1], dtype=np.int32)})
        #cluster_node_incidence_output.append({'ICp': np.array([0, 6, 12], dtype=np.int32),
        #                                      'ICi': np.array([0, 1, 2, 3, 6, 7, 4, 5, 8, 9, 10, 11], dtype=np.int32),
        #                                      'L': np.array([0, 1, 2, 3, 0, 1, 4, 5, 2, 3, 4, 5], dtype=np.int32)})
        #cluster_center_input[0] = {'a': [0, 1],
        #                           'num_clusters': 2,
        #                           'cm': np.array([0, 1, 1, 0, 0, 1], dtype=np.int32),
        #                           'ICp': np.array([0, 3, 6], dtype=np.int32),
        #                           'ICi': np.array([0, 3, 4, 1, 2, 5], dtype=np.int32),
        #                           'L': np.array([0, 0, 1, 1, 2, 2], dtype=np.int32)}
        #cluster_center_output[0] = [0, 1]

        ############################################################
        # (2) 16 node undirected, random length (0,2)
        ############################################################
        np.random.seed(2244369509)
        G.data[:] = np.random.rand(len(G.data)) * 2

        case = {}
        case['id'] = 2
        case['G'] = G
        case['input'] = {'centers': np.array([0, 1], dtype=np.int32)}
        cases_bellman_ford.append(case)
        cases_bellman_ford_balanced.append(case)

        ############################################################
        # (3) 16 node directed, random length
        ############################################################
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

        case = {}
        case['id'] = 3
        case['G'] = G
        case['input'] = {'centers': np.array([0, 5], np.int32)}
        cases_bellman_ford.append(case)
        cases_bellman_ford_balanced.append(case)

        ############################################################
        # (4) 191 node unstructured finite element matrix
        ############################################################
        G = load_example('unit_square')['A'].tocsr()
        G.data[:] = 1.0

        case = {}
        case['id'] = 4
        case['G'] = G
        case['input'] = {'centers': np.array([0, 10, 20, 30], dtype=np.int32)}
        cases_bellman_ford.append(case)
        cases_bellman_ford_balanced.append(case)

        ############################################################
        # (5) 5 nodes case in a line
        ############################################################
        Edges = np.array([[0, 1],
                          [1, 2],
                          [2, 3],
                          [3, 2],
                          [2, 1],
                          [1, 0]])
        w = np.array([1, 1, 1, 1, 1, 1], dtype=float)
        G = sparse.coo_matrix((w, (Edges[:, 0], Edges[:, 1])))
        G = G.tocsr()
        c = np.array([1,3], dtype=np.int32)

        case = {}
        case['id'] = 5
        case['G'] = G
        case['input'] = {'centers': c}
        cases_bellman_ford.append(case)
        cases_bellman_ford_balanced.append(case)

        ############################################################
        # (6) 5 node graph
        ############################################################
        Edges = np.array([[1, 4],
                          [3, 1],
                          [1, 3],
                          [0, 1],
                          [0, 2],
                          [3, 2],
                          [1, 2],
                          [4, 3]])
        w = np.array([2, 1, 2, 1, 4, 5, 3, 1], dtype=float)
        G = sparse.coo_matrix((w, (Edges[:, 0], Edges[:, 1])))
        G = G.tocsr()
        c = np.array([0,1,2], dtype=np.int32)

        case = {}
        case['id'] = 6
        case['G'] = G
        case['input'] = {'centers': c}
        cases_bellman_ford.append(case)
        cases_bellman_ford_balanced.append(case)

        self.cases_bellman_ford = cases_bellman_ford
        self.cases_bellman_ford_balanced = cases_bellman_ford_balanced

    def test_cluster_node_incidence(self):
        if 0:
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
        if 0:
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
        for case in self.cases_bellman_ford:
            G = case['G']

            if 'input' in case:
                centers = case['input']['centers']
                n = G.shape[0]

                d = np.empty(n, dtype=G.dtype)
                m = np.empty(n, dtype=np.int32)
                p = np.empty(n, dtype=np.int32)

                amg_core.bellman_ford(n, G.indptr, G.indices, G.data, centers,
                                      d, m, p, True)

            if 'output' in case:
                d_ref = case['output']['d']
                m_ref = case['output']['m']
            else:
                d_ref, m_ref, _ = bellman_ford_reference(G, centers)

            assert_array_equal(d, d_ref)
            assert_array_equal(m, m_ref)

    def test_bellman_ford_balanced(self):
        for case in self.cases_bellman_ford_balanced:
            G = case['G']

            if 'input' in case:
                centers = case['input']['centers']
                n = G.shape[0]

                d = np.empty(n, dtype=G.dtype)
                m = np.empty(n, dtype=np.int32)
                p = np.empty(n, dtype=np.int32)
                pc = np.empty(n, dtype=np.int32)
                s = np.empty(len(centers), dtype=np.int32)

                amg_core.bellman_ford_balanced(n, G.indptr, G.indices, G.data, centers,
                                               d, m, p, pc, s, True)

            if 'output' in case:
                d_ref = case['output']['d']
                m_ref = case['output']['m']
            else:
                d_ref, m_ref, _ = bellman_ford_balanced_reference(G, centers)

            assert_array_equal(d, d_ref)
            assert_array_equal(m, m_ref)

    def test_lloyd_cluster(self):
        if 0:
            centers = argin['centers']
            num_nodes = A.shape[0]
            c = centers.copy()

            d = np.full(num_nodes, np.inf, dtype=A.dtype)
            od = np.full(num_nodes, np.inf, dtype=A.dtype)
            d[centers] = 0
            od[centers] = 0

            m = np.full(num_nodes, -1, dtype=np.int32)
            m[c] = np.arange(len(c))

            p = np.full(num_nodes, -1, dtype=np.int32)

            amg_core.lloyd_cluster(num_nodes,
                                   A.indptr, A.indices, A.data,
                                   d, od, m, c, p)

            assert_array_equal(d, argout['d'])
            assert_array_equal(m, argout['m'])
            assert_array_equal(c, argout['c'])

    def test_lloyd_cluster_exact(self):
        if 0:
            centers = argin['centers']
            num_nodes = A.shape[0]
            c = centers.copy()
            num_clusters = len(centers)

            mv = np.finfo(A.dtype).max
            d = mv * np.ones(num_nodes, dtype=A.dtype)
            d[centers] = 0

            cm = -1 * np.ones(num_nodes, dtype=np.int32)
            cm[centers] = centers

            amg_core.lloyd_cluster_exact(num_nodes,
                                         A.indptr, A.indices, A.data,
                                         num_clusters,
                                         d, cm, c)

            assert_array_equal(d, argout['d'])
            assert_array_equal(cm, argout['cm'])
            assert_array_equal(c, argout['c'])
