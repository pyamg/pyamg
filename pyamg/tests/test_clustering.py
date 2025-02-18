"""Test clustering routines in amg_core."""
import numpy as np
from numpy.testing import TestCase, assert_array_equal
from scipy import sparse

from pyamg import amg_core
from pyamg.gallery import load_example
from pyamg.graph_ref import bellman_ford_reference, bellman_ford_balanced_reference


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
        G = sparse.csr_array(G)

        case = {}
        case['id'] = 0
        case['G'] = G
        case['input'] = {'centers': np.array([0, 5], dtype=np.int32)}
        case['output'] = {'m': np.array([0, 0, 1, 0, 0, 1], dtype=np.int32),
                          'd': np.array([0., 1., 1., 1., 1., 0.], dtype=G.dtype)}
        cases_bellman_ford.append(case)

        del case['output']
        cases_bellman_ford_balanced.append(case)

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
        G = sparse.csr_array(G)

        case = {}
        case['id'] = 1
        case['G'] = G
        case['input'] = {'centers': np.array([0, 1], dtype=np.int32)}
        cases_bellman_ford.append(case)
        cases_bellman_ford_balanced.append(case)

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
        G = sparse.csr_array(G)
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
        G.indptr.astype(np.int32, copy=False)
        G.indices.astype(np.int32, copy=False)

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
                          [1, 0]], dtype=np.int32)
        w = np.array([1, 1, 1, 1, 1, 1], dtype=float)
        G = sparse.coo_array((w, (Edges[:, 0], Edges[:, 1])))
        G = G.tocsr()
        c = np.array([1, 3], dtype=np.int32)

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
                          [4, 3]], dtype=np.int32)
        w = np.array([2, 1, 2, 1, 4, 5, 3, 1], dtype=float)
        G = sparse.coo_array((w, (Edges[:, 0], Edges[:, 1])))
        G = G.tocsr()
        c = np.array([0, 1, 2], dtype=np.int32)

        case = {}
        case['id'] = 6
        case['G'] = G
        case['input'] = {'centers': c}
        cases_bellman_ford.append(case)
        cases_bellman_ford_balanced.append(case)

        self.cases_bellman_ford = cases_bellman_ford
        self.cases_bellman_ford_balanced = cases_bellman_ford_balanced

    def test_bellman_ford(self):
        for case in self.cases_bellman_ford:
            G = case['G']

            if 'input' in case:
                centers = case['input']['centers']
                n = G.shape[0]

                d = np.full(n, np.inf, dtype=G.dtype)
                m = np.full(n, -1, dtype=np.int32)
                p = np.full(n, -1, dtype=np.int32)
                d[centers] = 0
                m[centers] = np.arange(len(centers))

                amg_core.bellman_ford(n, G.indptr, G.indices, G.data, centers,
                                      d, m, p)

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

                # initialize
                d = np.full(n, np.inf, dtype=G.dtype)         # distance to cluster center
                m = np.full(n, -1, dtype=np.int32)            # cluster membership or index
                p = np.full(n, -1, dtype=np.int32)            # predecessor on shortest path
                pc = np.full(n, 0, dtype=np.int32)            # predecessor count (0)
                s = np.full(len(centers), 1, dtype=np.int32)  # cluster size (1)
                d[centers] = 0                                # distance = 0 at centers
                m[centers] = np.arange(len(centers))          # number the membership

                print(case['id'])
                print(type(n))
                print(G.indptr.dtype)
                print(G.indices.dtype)
                print(G.data.dtype)
                amg_core.bellman_ford_balanced(n, G.indptr, G.indices, G.data, centers,
                                               d, m, p, pc, s, True)

            if 'output' in case:
                d_ref = case['output']['d']
                m_ref = case['output']['m']
            else:
                d_ref, m_ref, _ = bellman_ford_balanced_reference(G, centers)

            assert_array_equal(d, d_ref)
            assert_array_equal(m, m_ref)
