from numpy import ones, arange, array, diff, vstack
from scipy.sparse import csr_matrix

from pyamg.aggregation.aggregation import fit_candidates

from numpy.testing import TestCase, assert_almost_equal


class TestFitCandidates(TestCase):
    def setUp(self):
        self.cases = []

        # tests where AggOp includes all dofs
        # one candidate
        self.cases.append((
            csr_matrix((ones(5), array([0, 0, 0, 1, 1]), arange(6)),
                       shape=(5, 2)), ones((5, 1))))
        self.cases.append((
            csr_matrix((ones(5), array([1, 1, 0, 0, 0]), arange(6)),
                       shape=(5, 2)), ones((5, 1))))
        self.cases.append((
            csr_matrix((ones(9), array([0, 0, 0, 1, 1, 1, 2, 2, 2]),
                        arange(10)),
                       shape=(9, 3)), ones((9, 1))))
        self.cases.append((
            csr_matrix((ones(9), array([2, 1, 0, 0, 1, 2, 1, 0, 2]),
                        arange(10)),
                       shape=(9, 3)), arange(9).reshape(9, 1)))
        # two candidates
        self.cases.append((
            csr_matrix((ones(4), array([0, 0, 1, 1]), arange(5)),
                       shape=(4, 2)), vstack((ones(4), arange(4))).T))
        self.cases.append((
            csr_matrix((ones(9), array([0, 0, 0, 1, 1, 1, 2, 2, 2]),
                        arange(10)),
                       shape=(9, 3)), vstack((ones(9), arange(9))).T))
        self.cases.append((
            csr_matrix((ones(9), array([0, 0, 1, 1, 2, 2, 3, 3, 3]),
                        arange(10)),
                       shape=(9, 4)), vstack((ones(9), arange(9))).T))
        # two candidates, small norms
        self.cases.append((
            csr_matrix((ones(4), array([0, 0, 1, 1]), arange(5)),
                       shape=(4, 2)), vstack((ones(4), 1e-20 * arange(4))).T))
        self.cases.append((
            csr_matrix((ones(4), array([0, 0, 1, 1]), arange(5)),
                       shape=(4, 2)), 1e-20 * vstack((ones(4), arange(4))).T))

        # block aggregates, one candidate
        self.cases.append((
            csr_matrix((ones(3), array([0, 1, 1]), arange(4)),
                       shape=(3, 2)), ones((6, 1))))
        self.cases.append((
            csr_matrix((ones(3), array([0, 1, 1]), arange(4)),
                       shape=(3, 2)), ones((9, 1))))
        self.cases.append((
            csr_matrix((ones(5), array([2, 0, 2, 1, 1]), arange(6)),
                       shape=(5, 3)), ones((10, 1))))

        # block aggregates, two candidates
        self.cases.append((
            csr_matrix((ones(3), array([0, 1, 1]), arange(4)),
                       shape=(3, 2)), vstack((ones(6), arange(6))).T))
        self.cases.append((
            csr_matrix((ones(3), array([0, 1, 1]), arange(4)),
                       shape=(3, 2)), vstack((ones(9), arange(9))).T))
        self.cases.append((
            csr_matrix((ones(5), array([2, 0, 2, 1, 1]), arange(6)),
                       shape=(5, 3)), vstack((ones(10), arange(10))).T))

        # tests where AggOp excludes some dofs
        # one candidate
        self.cases.append((
            csr_matrix((ones(4), array([0, 0, 1, 1]),
                        array([0, 1, 2, 2, 3, 4])),
                       shape=(5, 2)), ones((5, 1))))
        self.cases.append((
            csr_matrix((ones(4), array([0, 0, 1, 1]),
                        array([0, 1, 2, 2, 3, 4])),
                       shape=(5, 2)), vstack((ones(5), arange(5))).T))

        # overdetermined blocks
        self.cases.append((
            csr_matrix((ones(4), array([0, 0, 1, 1]),
                        array([0, 1, 2, 2, 3, 4])),
                       shape=(5, 2)),
            vstack((ones(5), arange(5), arange(5)**2)).T))
        self.cases.append((
            csr_matrix((ones(6), array([1, 3, 0, 2, 1, 0]),
                        array([0, 0, 1, 2, 2, 3, 4, 5, 5, 6])),
                       shape=(9, 4)),
            vstack((ones(9), arange(9), arange(9) ** 2)).T))
        self.cases.append((
            csr_matrix((ones(6), array([1, 3, 0, 2, 1, 0]),
                        array([0, 0, 1, 2, 2, 3, 4, 5, 5, 6])),
                       shape=(9, 4)),
            vstack((ones(9), arange(9))).T))

        # complex tests
        # one aggregate one candidate
        # checks real part of complex
        self.cases.append((
            csr_matrix((ones(4), array([0, 0, 0, 0]), arange(5)),
                       shape=(4, 1)), (1 + 0j) * ones((4, 1))))
        self.cases.append((
            csr_matrix((ones(4), array([0, 0, 0, 0]), arange(5)),
                       shape=(4, 1)), (0 + 3j) * ones((4, 1))))
        # checks norm(), but not dot()
        self.cases.append((
            csr_matrix((ones(4), array([0, 0, 0, 0]), arange(5)),
                       shape=(4, 1)), (1 + 3j) * ones((4, 1))))
        # checks norm(), but not dot()
        self.cases.append((
            csr_matrix((ones(4), array([0, 0, 0, 0]), arange(5)),
                       shape=(4, 1)), (0 + 3j) * arange(4).reshape(4, 1)))
        # checks norm(), but not dot()
        self.cases.append((
            csr_matrix((ones(4), array([0, 0, 0, 0]), arange(5)),
                       shape=(4, 1)), (1 + 3j) * arange(4).reshape(4, 1)))
        self.cases.append((
            csr_matrix((ones(4), array([0, 0, 0, 0]), arange(5)),
                       shape=(4, 1)),
            array([[-1 + 4j], [0 + 5j], [5 - 2j], [9 - 8j]])))
        # one aggregate two candidates
        # checks real part of complex
        self.cases.append((
            csr_matrix((ones(4), array([0, 0, 0, 0]), arange(5)),
                       shape=(4, 1)),
            (1 + 0j) * vstack((ones(4), arange(4))).T))
        # checks norm() and dot()
        self.cases.append((
            csr_matrix((ones(4), array([0, 0, 0, 0]), arange(5)),
                       shape=(4, 1)),
            (1 + 3j) * vstack((ones(4), arange(4))).T))
        self.cases.append((
            csr_matrix((ones(4), array([0, 0, 0, 0]), arange(5)),
                       shape=(4, 1)),
            array([[-1 + 4j, 1 + 3j], [0 + 5j, 6 + 0j],
                   [5 - 2j, 7 + 1j], [9 - 8j, 7 + 2j]])))
        # two aggregates one candidates
        self.cases.append((
            csr_matrix((ones(4), array([0, 0, 1, 1]), arange(5)),
                       shape=(4, 2)), (1 + 3j) * arange(4).reshape(4, 1)))
        self.cases.append((
            csr_matrix((ones(4), array([0, 0, 1, 1]), arange(5)),
                       shape=(4, 2)), (0 + 3j) * arange(4).reshape(4, 1)))
        self.cases.append((
            csr_matrix((ones(4), array([0, 0, 1, 1]), arange(5)),
                       shape=(4, 2)), (1 + 3j) * arange(4).reshape(4, 1)))
        # two aggregates two candidates
        self.cases.append((
            csr_matrix((ones(4), array([0, 0, 1, 1]), arange(5)),
                       shape=(4, 2)),
            (1 + 0j) * vstack((ones(4), arange(4))).T))
        self.cases.append((
            csr_matrix((ones(4), array([0, 0, 1, 1]), arange(5)),
                       shape=(4, 2)),
            (0 + 3j) * vstack((ones(4), arange(4))).T))
        self.cases.append((
            csr_matrix((ones(4), array([0, 0, 1, 1]), arange(5)),
                       shape=(4, 2)),
            (1 + 3j) * vstack((ones(4), arange(4))).T))
        self.cases.append((
            csr_matrix((ones(4), array([0, 0, 1, 1]), arange(5)),
                       shape=(4, 2)),
            array([[-1 + 4j, 1 + 3j], [0 + 5j, 6 + 0j],
                   [5 - 2j, 7 + 1j], [9 - 8j, 7 + 2j]])))

    def test_all_cases(self):
        def mask_candidate(AggOp, candidates):
            # mask out all dofs that are not included in the aggregation
            candidates[diff(AggOp.indptr) == 0, :] = 0

        for AggOp, fine_candidates in self.cases:
            mask_candidate(AggOp, fine_candidates)

            Q, coarse_candidates = fit_candidates(AggOp, fine_candidates)

            # each fine level candidate should be fit (almost) exactly
            assert_almost_equal(fine_candidates, Q * coarse_candidates)
            assert_almost_equal(Q * (Q.H * fine_candidates), fine_candidates)
