from scipy.testing import *

from numpy import ones, arange, array, diff, vstack
from scipy.sparse import csr_matrix

from pyamg.aggregation.aggregation import fit_candidates


class TestFitCandidates(TestCase):
    def setUp(self):
        self.cases = []

        ### tests where AggOp includes all DOFs
        # one candidate
        self.cases.append((csr_matrix((ones(5),array([0,0,0,1,1]),arange(6)),shape=(5,2)), ones((5,1)) ))
        self.cases.append((csr_matrix((ones(5),array([1,1,0,0,0]),arange(6)),shape=(5,2)), ones((5,1)) ))
        self.cases.append((csr_matrix((ones(9),array([0,0,0,1,1,1,2,2,2]),arange(10)),shape=(9,3)), ones((9,1)) ))
        self.cases.append((csr_matrix((ones(9),array([2,1,0,0,1,2,1,0,2]),arange(10)),shape=(9,3)), arange(9).reshape(9,1) ))
        # two candidates
        self.cases.append((csr_matrix((ones(4),array([0,0,1,1]),arange(5)),shape=(4,2)), vstack((ones(4),arange(4))).T ))
        self.cases.append((csr_matrix((ones(9),array([0,0,0,1,1,1,2,2,2]),arange(10)),shape=(9,3)), vstack((ones(9),arange(9))).T ))
        self.cases.append((csr_matrix((ones(9),array([0,0,1,1,2,2,3,3,3]),arange(10)),shape=(9,4)), vstack((ones(9),arange(9))).T ))
       
        # block aggregates, one candidate
        self.cases.append((csr_matrix((ones(3),array([0,1,1]),arange(4)),shape=(3,2)), ones((6,1)) ))
        self.cases.append((csr_matrix((ones(3),array([0,1,1]),arange(4)),shape=(3,2)), ones((9,1)) ))
        self.cases.append((csr_matrix((ones(5),array([2,0,2,1,1]),arange(6)),shape=(5,3)), ones((10,1)) ))
        
        # block aggregates, two candidates
        self.cases.append((csr_matrix((ones(3),array([0,1,1]),arange(4)),shape=(3,2)), vstack((ones(6),arange(6))).T ))
        self.cases.append((csr_matrix((ones(3),array([0,1,1]),arange(4)),shape=(3,2)), vstack((ones(9),arange(9))).T ))
        self.cases.append((csr_matrix((ones(5),array([2,0,2,1,1]),arange(6)),shape=(5,3)), vstack((ones(10),arange(10))).T ))

        ### tests where AggOp excludes some DOFs
        # one candidate
        self.cases.append((csr_matrix((ones(4),array([0,0,1,1]),array([0,1,2,2,3,4])),shape=(5,2)), ones((5,1)) ))
        self.cases.append((csr_matrix((ones(4),array([0,0,1,1]),array([0,1,2,2,3,4])),shape=(5,2)), vstack((ones(5),arange(5))).T ))

        # overdetermined blocks
        self.cases.append((csr_matrix((ones(4),array([0,0,1,1]),array([0,1,2,2,3,4])),shape=(5,2)), vstack((ones(5),arange(5),arange(5)**2)).T  ))
        self.cases.append((csr_matrix((ones(6),array([1,3,0,2,1,0]),array([0,0,1,2,2,3,4,5,5,6])),shape=(9,4)), vstack((ones(9),arange(9),arange(9)**2)).T ))
        self.cases.append((csr_matrix((ones(6),array([1,3,0,2,1,0]),array([0,0,1,2,2,3,4,5,5,6])),shape=(9,4)), vstack((ones(9),arange(9))).T ))

    def test_all_cases(self):
        """Test case where aggregation includes all fine nodes"""

        def mask_candidate(AggOp,candidates):
            #mask out all DOFs that are not included in the aggregation
            candidates[diff(AggOp.indptr) == 0,:] = 0

        for AggOp,fine_candidates in self.cases:
            mask_candidate(AggOp,fine_candidates)

            Q,coarse_candidates = fit_candidates(AggOp,fine_candidates)

            #each fine level candidate should be fit (almost) exactly
            assert_almost_equal(fine_candidates,Q*coarse_candidates)
            assert_almost_equal(Q*(Q.T*fine_candidates),fine_candidates)


if __name__ == '__main__':
    nose.run(argv=['', __file__])

