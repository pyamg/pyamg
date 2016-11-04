import numpy as np
from scipy.sparse import csr_matrix, spdiags

from pyamg.gallery import poisson, load_example
from pyamg.strength import symmetric_strength_of_connection
from pyamg.aggregation.aggregate import standard_aggregation, naive_aggregation

from numpy.testing import TestCase, assert_equal


class TestAggregate(TestCase):
    def setUp(self):
        self.cases = []

        # random matrices
        np.random.seed(0)
        for N in [2, 3, 5]:
            self.cases.append(csr_matrix(np.random.rand(N, N)))

        # Poisson problems in 1D and 2D
        for N in [2, 3, 5, 7, 10, 11, 19]:
            self.cases.append(poisson((N,), format='csr'))
        for N in [2, 3, 5, 7, 8]:
            self.cases.append(poisson((N, N), format='csr'))

        for name in ['knot', 'airfoil', 'bar']:
            ex = load_example(name)
            self.cases.append(ex['A'].tocsr())

    def test_standard_aggregation(self):
        for A in self.cases:
            S = symmetric_strength_of_connection(A)

            (expected, expected_Cpts) = reference_standard_aggregation(S)
            (result, Cpts) = standard_aggregation(S)

            assert_equal((result - expected).nnz, 0)
            assert_equal(Cpts.shape[0], expected_Cpts.shape[0])
            assert_equal(np.setdiff1d(Cpts, expected_Cpts).shape[0], 0)

        # S is diagonal - no dofs aggregated
        S = spdiags([[1, 1, 1, 1]], [0], 4, 4, format='csr')
        (result, Cpts) = standard_aggregation(S)
        expected = np.array([[0], [0], [0], [0]])
        assert_equal(result.todense(), expected)
        assert_equal(Cpts.shape[0], 0)

    def test_naive_aggregation(self):
        for A in self.cases:
            S = symmetric_strength_of_connection(A)

            (expected, expected_Cpts) = reference_naive_aggregation(S)
            (result, Cpts) = naive_aggregation(S)

            assert_equal((result - expected).nnz, 0)
            assert_equal(Cpts.shape[0], expected_Cpts.shape[0])
            assert_equal(np.setdiff1d(Cpts, expected_Cpts).shape[0], 0)

        # S is diagonal - no dofs aggregated
        S = spdiags([[1, 1, 1, 1]], [0], 4, 4, format='csr')
        (result, Cpts) = naive_aggregation(S)
        expected = np.eye(4)
        assert_equal(result.todense(), expected)
        assert_equal(Cpts.shape[0], 4)


class TestComplexAggregate(TestCase):
    def setUp(self):
        self.cases = []

        # Poisson problems in 2D
        for N in [2, 3, 5, 7, 8]:
            A = poisson((N, N), format='csr')
            A.data = A.data + 0.001j*np.random.rand(A.nnz)
            self.cases.append(A)

    def test_standard_aggregation(self):
        for A in self.cases:
            S = symmetric_strength_of_connection(A)

            (expected, expected_Cpts) = reference_standard_aggregation(S)
            (result, Cpts) = standard_aggregation(S)

            assert_equal((result - expected).nnz, 0)
            assert_equal(Cpts.shape[0], expected_Cpts.shape[0])
            assert_equal(np.setdiff1d(Cpts, expected_Cpts).shape[0], 0)

    def test_naive_aggregation(self):
        for A in self.cases:
            S = symmetric_strength_of_connection(A)

            (expected, expected_Cpts) = reference_naive_aggregation(S)
            (result, Cpts) = naive_aggregation(S)

            assert_equal((result - expected).nnz, 0)
            assert_equal(Cpts.shape[0], expected_Cpts.shape[0])
            assert_equal(np.setdiff1d(Cpts, expected_Cpts).shape[0], 0)


# reference implementations for unittests  #
# note that this method only tests the current implementation, not
# all possible implementations
def reference_standard_aggregation(C):
    S = np.array_split(C.indices, C.indptr[1:-1])

    n = C.shape[0]

    R = set(range(n))
    j = 0
    Cpts = []

    aggregates = np.empty(n, dtype=C.indices.dtype)
    aggregates[:] = -1

    # Pass #1
    for i, row in enumerate(S):
        Ni = set(row) | set([i])

        if Ni.issubset(R):
            Cpts.append(i)
            R -= Ni
            for x in Ni:
                aggregates[x] = j
            j += 1

    # Pass #2
    Old_R = R.copy()
    for i, row in enumerate(S):
        if i not in R:
            continue

        for x in row:
            if x not in Old_R:
                aggregates[i] = aggregates[x]
                R.remove(i)
                break

    # Pass #3
    for i, row in enumerate(S):
        if i not in R:
            continue
        Ni = set(row) | set([i])
        Cpts.append(i)

        for x in Ni:
            if x in R:
                aggregates[x] = j
            j += 1

    assert(len(R) == 0)

    Pj = aggregates
    Pp = np.arange(n+1)
    Px = np.ones(n)

    return csr_matrix((Px, Pj, Pp)), np.array(Cpts)


def reference_naive_aggregation(C):
    S = np.array_split(C.indices, C.indptr[1:-1])
    n = C.shape[0]
    aggregates = np.empty(n, dtype=C.indices.dtype)
    aggregates[:] = -1  # aggregates[j] denotes the aggregate j is in
    R = np.zeros((0,))     # R stores already aggregated nodes
    j = 0               # j is the aggregate counter
    Cpts = []

    # Only one aggregation pass
    for i, row in enumerate(S):

        # if i isn't already aggregated, grab all his neighbors
        if aggregates[i] == -1:
            unaggregated_neighbors = np.setdiff1d(row, R)
            aggregates[unaggregated_neighbors] = j
            aggregates[i] = j
            j += 1
            R = np.union1d(R, unaggregated_neighbors)
            R = np.union1d(R, np.array([i]))
            Cpts.append(i)
        else:
            pass

    assert(np.unique(R).shape[0] == n)

    Pj = aggregates
    Pp = np.arange(n+1)
    Px = np.ones(n)

    return csr_matrix((Px, Pj, Pp)), np.array(Cpts)
