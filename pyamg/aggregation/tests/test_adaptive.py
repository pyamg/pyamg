import numpy as np
import scipy as sp

from pyamg.gallery import poisson, linear_elasticity
from pyamg.aggregation import smoothed_aggregation_solver
from pyamg.aggregation.adaptive import adaptive_sa_solver

from numpy.testing import TestCase

import warnings
from scipy.sparse import SparseEfficiencyWarning
warnings.simplefilter('ignore', SparseEfficiencyWarning)
warnings.filterwarnings('ignore', category=UserWarning,
                        message='Having less target vectors')


class TestAdaptiveSA(TestCase):
    def setUp(self):
        np.random.seed(0)

    def test_poisson(self):
        A = poisson((50, 50), format='csr')

        [asa, work] = adaptive_sa_solver(A, num_candidates=1)
        sa = smoothed_aggregation_solver(A, B=np.ones((A.shape[0], 1)))

        b = sp.rand(A.shape[0])

        residuals0 = []
        residuals1 = []

        sol0 = asa.solve(b, maxiter=20, tol=1e-10, residuals=residuals0)
        sol1 = sa.solve(b, maxiter=20, tol=1e-10, residuals=residuals1)
        del sol0, sol1

        conv_asa = (residuals0[-1] / residuals0[0]) ** (1.0 / len(residuals0))
        conv_sa = (residuals1[-1] / residuals1[0]) ** (1.0 / len(residuals1))

        # print "ASA convergence (Poisson)",conv_asa
        # print "SA convergence (Poisson)",conv_sa
        assert(conv_asa < 1.2 * conv_sa)

    def test_elasticity(self):
        A, B = linear_elasticity((35, 35), format='bsr')

        smoother = ('gauss_seidel', {'sweep': 'symmetric', 'iterations': 2})
        [asa, work] = adaptive_sa_solver(A, num_candidates=3,
                                         improvement_iters=5,
                                         prepostsmoother=smoother)
        sa = smoothed_aggregation_solver(A, B=B)

        b = sp.rand(A.shape[0])

        residuals0 = []
        residuals1 = []

        sol0 = asa.solve(b, maxiter=20, tol=1e-10, residuals=residuals0)
        sol1 = sa.solve(b, maxiter=20, tol=1e-10, residuals=residuals1)
        del sol0, sol1

        conv_asa = (residuals0[-1] / residuals0[0]) ** (1.0 / len(residuals0))
        conv_sa = (residuals1[-1] / residuals1[0]) ** (1.0 / len(residuals1))

        # print "ASA convergence (Elasticity) %1.2e" % (conv_asa)
        # print "SA convergence (Elasticity) %1.2e" % (conv_sa)
        assert(conv_asa < 1.3 * conv_sa)

    def test_matrix_formats(self):

        # Do dense, csr, bsr and csc versions of A all yield the same solver
        A = poisson((7, 7), format='csr')
        cases = [A.tobsr(blocksize=(1, 1))]
        cases.append(A.tocsc())
        cases.append(A.todense())
        warnings.filterwarnings('ignore', message='SparseEfficiencyWarning')

        np.random.seed(0)
        sa_old = adaptive_sa_solver(A, initial_candidates=np.ones((49, 1)),
                                    max_coarse=10)[0]
        for AA in cases:
            np.random.seed(0)
            sa_new = adaptive_sa_solver(AA,
                                        initial_candidates=np.ones((49, 1)),
                                        max_coarse=10)[0]
            assert(abs(np.ravel(sa_old.levels[-1].A.todense() -
                                sa_new.levels[-1].A.todense())).max() < 0.01)
            sa_old = sa_new


class TestComplexAdaptiveSA(TestCase):
    def setUp(self):
        np.random.seed(0)

    def test_poisson(self):
        cases = []

        # perturbed Laplacian
        A = poisson((50, 50), format='csr')
        Ai = A.copy()
        Ai.data = Ai.data + 1e-5j * sp.rand(Ai.nnz)
        cases.append((Ai, 0.25))

        # imaginary Laplacian
        Ai = 1.0j * A
        cases.append((Ai, 0.25))

        # JBS:  Not sure if this is a valid test case
        # imaginary shift
        # Ai = A + 1.1j*scipy.sparse.eye(A.shape[0], A.shape[1])
        # cases.append((Ai,0.8))

        for A, rratio in cases:
            [asa, work] = adaptive_sa_solver(A, num_candidates=1,
                                             symmetry='symmetric')
            # sa = smoothed_aggregation_solver(A, B = np.ones((A.shape[0],1)) )

            b = np.zeros((A.shape[0],))
            x0 = sp.rand(A.shape[0],) + 1.0j * sp.rand(A.shape[0],)

            residuals0 = []

            sol0 = asa.solve(b, x0=x0, maxiter=20, tol=1e-10,
                             residuals=residuals0)
            del sol0

            conv_asa = \
                (residuals0[-1] / residuals0[0]) ** (1.0 / len(residuals0))

            assert(conv_asa < rratio)

# class TestAugmentCandidates(TestCase):
#    def setUp(self):
#        self.cases = []
#
# two candidates
#
# block candidates
# self.cases.append((
#   csr_matrix((np.ones(9),array([0,0,0,1,1,1,2,2,2]),arange(10)),
#   shape=(9,3)), vstack((array([1]*9 + [0]*9),arange(2*9))).T ))
#
#    def test_first_level(self):
#        cases = []
#
# tests where AggOp includes all DOFs
#        cases.append((
#           csr_matrix((np.ones(4),array([0,0,1,1]),arange(5)),
#           shape=(4,2)), vstack((np.ones(4),arange(4))).T ))
#        cases.append((
#           csr_matrix((np.ones(9),array([0,0,0,1,1,1,2,2,2]),arange(10)),
#           shape=(9,3)), vstack((np.ones(9),arange(9))).T ))
#        cases.append((
#           csr_matrix((np.ones(9),array([0,0,1,1,2,2,3,3,3]),arange(10)),
#           shape=(9,4)), vstack((np.ones(9),arange(9))).T ))
#
# tests where AggOp excludes some DOFs
#        cases.append((
#           csr_matrix((np.ones(4),array([0,0,1,1]),array([0,1,2,2,3,4])),
#           shape=(5,2)), vstack((np.ones(5),arange(5))).T ))
#
# overdetermined blocks
#        cases.append((
#           csr_matrix((np.ones(4),array([0,0,1,1]),array([0,1,2,2,3,4])),
#           shape=(5,2)), vstack((np.ones(5),arange(5),arange(5)**2)).T  ))
#        cases.append((
#           csr_matrix(
#               (np.ones(6),array([1,3,0,2,1,0]),array([0,0,1,2,2,3,4,5,5,6])),
#           shape=(9,4)), vstack((np.ones(9),arange(9),arange(9)**2)).T ))
#        cases.append((
#           csr_matrix(
#               (np.ones(6),array([1,3,0,2,1,0]),array([0,0,1,2,2,3,4,5,5,6])),
#           shape=(9,4)), vstack((np.ones(9),arange(9))).T ))
#
#        def mask_candidate(AggOp,candidates):
# mask out all DOFs that are not included in the aggregation
#            candidates[diff(AggOp.indptr) == 0,:] = 0
#
#        for AggOp,fine_candidates in cases:
#
#            mask_candidate(AggOp,fine_candidates)
#
#            for i in range(1,fine_candidates.shape[1]):
#                Q_expected,R_expected =
#                   fit_candidates(AggOp,fine_candidates[:, :i+1])
#
#                old_Q, old_R = fit_candidates(AggOp,fine_candidates[:,:i])
#
#                Q_result,R_result = augment_candidates(AggOp, old_Q, old_R,
#                   fine_candidates[:, [i]])
#
# compare against SA method (which is assumed to be correct)
#                assert_almost_equal(Q_expected.todense(),Q_result.todense())
#                assert_almost_equal(R_expected,R_result)
#
# each fine level candidate should be fit exactly
#                assert_almost_equal(fine_candidates[:,:i+1],Q_result*R_result)
#                assert_almost_equal(
#                   Q_result*(Q_result.T*fine_candidates[:, :i+1]),
#                   fine_candidates[:, :i+1])
