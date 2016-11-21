import numpy as np
from scipy.sparse import csr_matrix

from pyamg.aggregation.aggregation import fit_candidates

from numpy.testing import TestCase, assert_almost_equal


class TestFitCandidates(TestCase):
    def setUp(self):
        self.cases = []

        # tests where AggOp includes all dofs
        # one candidate
        self.cases.append((
            csr_matrix((np.ones(5), np.array([0, 0, 0, 1, 1]), np.arange(6)),
                       shape=(5, 2)), np.ones((5, 1))))
        self.cases.append((
            csr_matrix((np.ones(5), np.array([1, 1, 0, 0, 0]), np.arange(6)),
                       shape=(5, 2)), np.ones((5, 1))))
        self.cases.append((
            csr_matrix((np.ones(9), np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]),
                        np.arange(10)),
                       shape=(9, 3)), np.ones((9, 1))))
        self.cases.append((
            csr_matrix((np.ones(9), np.array([2, 1, 0, 0, 1, 2, 1, 0, 2]),
                        np.arange(10)),
                       shape=(9, 3)), np.arange(9).reshape(9, 1)))
        # two candidates
        self.cases.append((
            csr_matrix((np.ones(4), np.array([0, 0, 1, 1]), np.arange(5)),
                       shape=(4, 2)), np.vstack((np.ones(4), np.arange(4))).T))
        self.cases.append((
            csr_matrix((np.ones(9), np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]),
                        np.arange(10)),
                       shape=(9, 3)), np.vstack((np.ones(9), np.arange(9))).T))
        self.cases.append((
            csr_matrix((np.ones(9), np.array([0, 0, 1, 1, 2, 2, 3, 3, 3]),
                        np.arange(10)),
                       shape=(9, 4)), np.vstack((np.ones(9), np.arange(9))).T))
        # two candidates, small norms
        self.cases.append((
            csr_matrix((np.ones(4), np.array([0, 0, 1, 1]), np.arange(5)),
                       shape=(4, 2)),
            np.vstack((np.ones(4), 1e-20 * np.arange(4))).T))
        self.cases.append((
            csr_matrix((np.ones(4), np.array([0, 0, 1, 1]), np.arange(5)),
                       shape=(4, 2)),
            1e-20 * np.vstack((np.ones(4), np.arange(4))).T))

        # block aggregates, one candidate
        self.cases.append((
            csr_matrix((np.ones(3), np.array([0, 1, 1]), np.arange(4)),
                       shape=(3, 2)), np.ones((6, 1))))
        self.cases.append((
            csr_matrix((np.ones(3), np.array([0, 1, 1]), np.arange(4)),
                       shape=(3, 2)), np.ones((9, 1))))
        self.cases.append((
            csr_matrix((np.ones(5), np.array([2, 0, 2, 1, 1]), np.arange(6)),
                       shape=(5, 3)), np.ones((10, 1))))

        # block aggregates, two candidates
        self.cases.append((
            csr_matrix((np.ones(3), np.array([0, 1, 1]), np.arange(4)),
                       shape=(3, 2)), np.vstack((np.ones(6), np.arange(6))).T))
        self.cases.append((
            csr_matrix((np.ones(3), np.array([0, 1, 1]), np.arange(4)),
                       shape=(3, 2)), np.vstack((np.ones(9), np.arange(9))).T))
        self.cases.append((
            csr_matrix((np.ones(5), np.array([2, 0, 2, 1, 1]), np.arange(6)),
                       shape=(5, 3)),
            np.vstack((np.ones(10), np.arange(10))).T))

        # tests where AggOp excludes some dofs
        # one candidate
        self.cases.append((
            csr_matrix((np.ones(4), np.array([0, 0, 1, 1]),
                        np.array([0, 1, 2, 2, 3, 4])),
                       shape=(5, 2)), np.ones((5, 1))))
        self.cases.append((
            csr_matrix((np.ones(4), np.array([0, 0, 1, 1]),
                        np.array([0, 1, 2, 2, 3, 4])),
                       shape=(5, 2)), np.vstack((np.ones(5), np.arange(5))).T))

        # overdetermined blocks
        self.cases.append((
            csr_matrix((np.ones(4), np.array([0, 0, 1, 1]),
                        np.array([0, 1, 2, 2, 3, 4])),
                       shape=(5, 2)),
            np.vstack((np.ones(5), np.arange(5), np.arange(5)**2)).T))
        self.cases.append((
            csr_matrix((np.ones(6), np.array([1, 3, 0, 2, 1, 0]),
                        np.array([0, 0, 1, 2, 2, 3, 4, 5, 5, 6])),
                       shape=(9, 4)),
            np.vstack((np.ones(9), np.arange(9), np.arange(9) ** 2)).T))
        self.cases.append((
            csr_matrix((np.ones(6), np.array([1, 3, 0, 2, 1, 0]),
                        np.array([0, 0, 1, 2, 2, 3, 4, 5, 5, 6])),
                       shape=(9, 4)),
            np.vstack((np.ones(9), np.arange(9))).T))

        # complex tests
        # one aggregate one candidate
        # checks real part of complex
        self.cases.append((
            csr_matrix((np.ones(4), np.array([0, 0, 0, 0]), np.arange(5)),
                       shape=(4, 1)), (1 + 0j) * np.ones((4, 1))))
        self.cases.append((
            csr_matrix((np.ones(4), np.array([0, 0, 0, 0]), np.arange(5)),
                       shape=(4, 1)), (0 + 3j) * np.ones((4, 1))))
        # checks norm(), but not dot()
        self.cases.append((
            csr_matrix((np.ones(4), np.array([0, 0, 0, 0]), np.arange(5)),
                       shape=(4, 1)), (1 + 3j) * np.ones((4, 1))))
        # checks norm(), but not dot()
        self.cases.append((
            csr_matrix((np.ones(4), np.array([0, 0, 0, 0]), np.arange(5)),
                       shape=(4, 1)), (0 + 3j) * np.arange(4).reshape(4, 1)))
        # checks norm(), but not dot()
        self.cases.append((
            csr_matrix((np.ones(4), np.array([0, 0, 0, 0]), np.arange(5)),
                       shape=(4, 1)), (1 + 3j) * np.arange(4).reshape(4, 1)))
        self.cases.append((
            csr_matrix((np.ones(4), np.array([0, 0, 0, 0]), np.arange(5)),
                       shape=(4, 1)),
            np.array([[-1 + 4j], [0 + 5j], [5 - 2j], [9 - 8j]])))
        # one aggregate two candidates
        # checks real part of complex
        self.cases.append((
            csr_matrix((np.ones(4), np.array([0, 0, 0, 0]), np.arange(5)),
                       shape=(4, 1)),
            (1 + 0j) * np.vstack((np.ones(4), np.arange(4))).T))
        # checks norm() and dot()
        self.cases.append((
            csr_matrix((np.ones(4), np.array([0, 0, 0, 0]), np.arange(5)),
                       shape=(4, 1)),
            (1 + 3j) * np.vstack((np.ones(4), np.arange(4))).T))
        self.cases.append((
            csr_matrix((np.ones(4), np.array([0, 0, 0, 0]), np.arange(5)),
                       shape=(4, 1)),
            np.array([[-1 + 4j, 1 + 3j], [0 + 5j, 6 + 0j],
                      [5 - 2j, 7 + 1j], [9 - 8j, 7 + 2j]])))
        # two aggregates one candidates
        self.cases.append((
            csr_matrix((np.ones(4), np.array([0, 0, 1, 1]), np.arange(5)),
                       shape=(4, 2)), (1 + 3j) * np.arange(4).reshape(4, 1)))
        self.cases.append((
            csr_matrix((np.ones(4), np.array([0, 0, 1, 1]), np.arange(5)),
                       shape=(4, 2)), (0 + 3j) * np.arange(4).reshape(4, 1)))
        self.cases.append((
            csr_matrix((np.ones(4), np.array([0, 0, 1, 1]), np.arange(5)),
                       shape=(4, 2)), (1 + 3j) * np.arange(4).reshape(4, 1)))
        # two aggregates two candidates
        self.cases.append((
            csr_matrix((np.ones(4), np.array([0, 0, 1, 1]), np.arange(5)),
                       shape=(4, 2)),
            (1 + 0j) * np.vstack((np.ones(4), np.arange(4))).T))
        self.cases.append((
            csr_matrix((np.ones(4), np.array([0, 0, 1, 1]), np.arange(5)),
                       shape=(4, 2)),
            (0 + 3j) * np.vstack((np.ones(4), np.arange(4))).T))
        self.cases.append((
            csr_matrix((np.ones(4), np.array([0, 0, 1, 1]), np.arange(5)),
                       shape=(4, 2)),
            (1 + 3j) * np.vstack((np.ones(4), np.arange(4))).T))
        self.cases.append((
            csr_matrix((np.ones(4), np.array([0, 0, 1, 1]), np.arange(5)),
                       shape=(4, 2)),
            np.array([[-1 + 4j, 1 + 3j], [0 + 5j, 6 + 0j],
                      [5 - 2j, 7 + 1j], [9 - 8j, 7 + 2j]])))

    def test_all_cases(self):
        def mask_candidate(AggOp, candidates):
            # mask out all dofs that are not included in the aggregation
            candidates[np.where(np.diff(AggOp.indptr) == 0)[0], :] = 0

        for AggOp, fine_candidates in self.cases:
            mask_candidate(AggOp, fine_candidates)

            Q, coarse_candidates = fit_candidates(AggOp, fine_candidates)

            # each fine level candidate should be fit (almost) exactly
            assert_almost_equal(fine_candidates, Q * coarse_candidates)
            assert_almost_equal(Q * (Q.H * fine_candidates), fine_candidates)
