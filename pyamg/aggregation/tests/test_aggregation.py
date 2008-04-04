from scipy.testing import *

import numpy
from numpy import sqrt, ones, arange, array, diff, vstack
from scipy import rand
from scipy.sparse import csr_matrix

from pyamg.utils import diag_sparse
from pyamg.gallery import poisson, linear_elasticity

from pyamg.aggregation.aggregation import smoothed_aggregation_solver, fit_candidates



#    def test_user_aggregation(self):
#        """check that the sa_interpolation accepts user-defined aggregates"""
#
#        user_cases = []
#
#        #simple 1d example w/ two aggregates
#        A = poisson( (6,), format='csr')
#        AggOp = csr_matrix((ones(6),array([0,0,0,1,1,1]),arange(7)),shape=(6,2))
#        candidates = ones((6,1))
#        user_cases.append((A,AggOp,candidates))
#
#        #simple 1d example w/ two aggregates (not all nodes are aggregated)
#        A = poisson( (6,), format='csr')
#        AggOp = csr_matrix((ones(4),array([0,0,1,1]),array([0,1,1,2,3,3,4])),shape=(6,2))
#        candidates = ones((6,1))
#        user_cases.append((A,AggOp,candidates))
#
#        for A,AggOp,candidates in user_cases:
#            T,coarse_candidates_result = fit_candidates(AggOp,candidates)
#
#            P_result = sa_interpolation(A,candidates,omega=4.0/3.0,AggOp=AggOp)[0]
#            P_expected = jacobi_prolongation_smoother(A, T, omega=4.0/3.0)
#
#            assert_almost_equal(P_result.todense(),P_expected.todense())



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

            #each fine level candidate should be fit exactly
            assert_almost_equal(fine_candidates,Q*coarse_candidates)
            assert_almost_equal(Q*(Q.T*fine_candidates),fine_candidates)


class TestSASolverPerformance(TestCase):
    def setUp(self):
        self.cases = []

        self.cases.append(( poisson( (10000,),  format='csr'), None))
        self.cases.append(( poisson( (100,100), format='csr'), None))
        self.cases.append( linear_elasticity( (100,100), format='bsr') )
        # TODO add unstructured tests


    def test_basic(self):
        """check that method converges at a reasonable rate"""

        for A,B in self.cases:
            ml = smoothed_aggregation_solver(A, B, max_coarse=10)

            numpy.random.seed(0) #make tests repeatable

            x = rand(A.shape[0])
            b = A*rand(A.shape[0])

            x_sol,residuals = ml.solve(b,x0=x,maxiter=20,tol=1e-10,return_residuals=True)

            avg_convergence_ratio = (residuals[-1]/residuals[0])**(1.0/len(residuals))
            
            assert(avg_convergence_ratio < 0.3)

    def test_DAD(self):
        A = poisson( (50,50), format='csr' )        

        x = rand(A.shape[0])
        b = rand(A.shape[0])
 
        D     = diag_sparse(1.0/sqrt(10**(12*rand(A.shape[0])-6))).tocsr()
        D_inv = diag_sparse(1.0/D.data)
 
        DAD   = D*A*D
 
        B = ones((A.shape[0],1))
 
        #TODO force 2 level method and check that result is the same
 
        sa = smoothed_aggregation_solver(D*A*D, D_inv * B, max_coarse=1, max_levels=2)
 
        x_sol,residuals = sa.solve(b,x0=x,maxiter=10,tol=1e-12,return_residuals=True)
 
        avg_convergence_ratio = (residuals[-1]/residuals[0])**(1.0/len(residuals))
        
        assert(avg_convergence_ratio < 0.25)



if __name__ == '__main__':
    nose.run(argv=['', __file__])
