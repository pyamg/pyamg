"""Test aggregation methods."""
import numpy as np
from numpy.testing import TestCase, assert_equal
from scipy import sparse

from pyamg.gallery import poisson, load_example
from pyamg.strength import (symmetric_strength_of_connection,
                            classical_strength_of_connection)
from pyamg.aggregation.aggregate import (standard_aggregation, naive_aggregation,
                                         pairwise_aggregation)

from collections import OrderedDict


class TestAggregate(TestCase):
    def setUp(self):
        self.cases = []

        # random matrices
        np.random.seed(2006482792)
        for N in [2, 3, 5]:
            self.cases.append(sparse.csr_matrix(np.random.rand(N, N)))

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
        S = sparse.spdiags([[1, 1, 1, 1]], [0], 4, 4, format='csr')
        (result, Cpts) = standard_aggregation(S)
        expected = np.array([[0], [0], [0], [0]])
        assert_equal(result.toarray(), expected)
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
        S = sparse.spdiags([[1, 1, 1, 1]], [0], 4, 4, format='csr')
        (result, Cpts) = naive_aggregation(S)
        expected = np.eye(4)
        assert_equal(result.toarray(), expected)
        assert_equal(Cpts.shape[0], 4)

    def test_pairwise_aggregation(self):
        for A in self.cases:
            S = classical_strength_of_connection(A, theta=0.25, norm='min')
            (expected, expected_Cpts) = reference_pairwise_aggregation(S)
            (result, Cpts) = pairwise_aggregation(A, matchings=1, theta=0.25, norm='min')

            assert_equal((result - expected).nnz, 0)
            assert_equal(Cpts.shape[0], expected_Cpts.shape[0])
            assert_equal(np.setdiff1d(Cpts, expected_Cpts).shape[0], 0)

        # S is diagonal - no dofs aggregated
        S = sparse.spdiags([[1.0, 1.0, 1.0, 1.0]], [0], 4, 4, format='csr')
        (result, Cpts) = pairwise_aggregation(S, matchings=1, theta=0.25, norm='min')
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
        Ni = set(row) | {i}

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
        Ni = set(row) | {i}
        Cpts.append(i)

        for x in Ni:
            if x in R:
                aggregates[x] = j
            j += 1

    assert len(R) == 0

    Pj = aggregates
    Pp = np.arange(n+1)
    Px = np.ones(n)

    return sparse.csr_matrix((Px, Pj, Pp)), np.array(Cpts)


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

    assert np.unique(R).shape[0] == n

    Pj = aggregates
    Pp = np.arange(n+1)
    Px = np.ones(n)

    return sparse.csr_matrix((Px, Pj, Pp)), np.array(Cpts)


def reference_pairwise_aggregation(C):
    S = np.array_split(C.indices, C.indptr[1:-1])
    data = np.array_split(C.data, C.indptr[1:-1])
    n = C.shape[0]
    aggregates = np.empty(n, dtype=C.indices.dtype)
    aggregates[:] = -1     # aggregates[j] denotes the aggregate j is in
    R = np.zeros((0,))     # R stores already aggregated nodes
    aggregate_count = 0               # aggregate_count is the aggregate counter
    Cpts = []
    m = np.zeros(n, dtype=C.indices.dtype)
    for row in S:
        m[row] = m[row] + 1

    max_m = max(m)

    # using a ordereddict here as there's no orderedset in python
    mmap = [OrderedDict() for _ in range(0, max_m+1)]

    for i in range(n):
        mmap[m[i]][i] = True

    count = 0
    aggregate_count = 0
    while (count < n):
        for k in range(0, max_m+1):
            if mmap[k]:
                i = list(mmap[k].keys())[0]
                break

        row = S[i]
        R = np.union1d(R, np.array([i]))
        aggregate_set = [i]
        aggregates[i] = aggregate_count
        max_aij = -np.inf
        max_aij_index = -1

        for k, j in enumerate(row):
            if j not in R and data[i][k] >= max_aij:
                max_aij = data[i][k]
                max_aij_index = j

        if max_aij_index != -1:
            j = max_aij_index
            aggregate_set.append(j)
            R = np.union1d(R, np.array([j]))
            aggregates[j] = aggregate_count

        Cpts.append(i)

        # Remove the aggregated nodes from mmap
        for j in aggregate_set:
            del mmap[m[j]][j]

        # Reduce m of the neighbors of the aggregated nodes
        for j in aggregate_set:
            row = S[j]
            unaggregated_neighbors = np.setdiff1d(row, R)
            for neighbour in unaggregated_neighbors:
                del mmap[m[neighbour]][neighbour]
                m[neighbour] = m[neighbour] - 1
                mmap[m[neighbour]][neighbour] = True

        count += len(aggregate_set)
        aggregate_count += 1

    assert (np.unique(R).shape[0] == n)

    Pj = aggregates
    Pp = np.arange(n+1)
    Px = np.ones(n)

    return sparse.csr_matrix((Px, Pj, Pp)), np.array(Cpts)
