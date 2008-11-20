from pyamg.testing import *
import scipy.sparse
import numpy

from numpy import array, zeros, ones, mat
from scipy import rand
from scipy.linalg import pinv2
from scipy.sparse import bsr_matrix

from pyamg.aggregation.smooth import Satisfy_Constraints
from pyamg.gallery import poisson, linear_elasticity, load_example, gauge_laplacian
from pyamg.aggregation import smoothed_aggregation_solver

class TestEnergyMin(TestCase):
    def test_range(self):
        """Check that P*R=B"""
        numpy.random.seed(0) #make tests repeatable

        cases = []
        A = poisson((10,10), format='csr')
        B = ones((A.shape[0],1))
        cases.append((A,B,'energy'))
        cases.append((A,B,('energy', {'maxiter' : 3}) ))
        cases.append((A,B,('energy', {'SPD' : False}) ))
        cases.append((A,B,('energy', {'degree' : 2}) ))
        cases.append((A,B,('energy', {'SPD' : False, 'degree' : 2}) ))
        
        iA = 1.0j*A
        iB = 1.0 + rand(iA.shape[0],2) + 1.0j*(1.0 + rand(iA.shape[0],2))
        cases.append((iA,B,('energy', {'SPD' : False}) ))
        cases.append((iA,B,('energy', {'SPD' : False, 'degree' : 2}) ))
        cases.append((iA.tobsr(blocksize=(5,5)),B,('energy', {'SPD' : False, 'degree' : 2, 'maxiter' : 3}) ))
        cases.append((iA,iB,('energy', {'SPD' : False}) ))
        cases.append((iA.tobsr(blocksize=(5,5)),iB,('energy', {'SPD' : False}) ))
        cases.append((iA,iB,('energy', {'SPD' : False, 'degree' : 2}) ))
        cases.append((iA,iB,('energy', {'SPD' : False, 'degree' : 2, 'maxiter' : 3}) ))
 
        iA = A + 1.0j*scipy.sparse.eye(A.shape[0], A.shape[1]);
        cases.append((iA,B,('energy', {'SPD' : False}) ))
        cases.append((iA,B,('energy', {'SPD' : False, 'degree' : 2}) ))
        cases.append((iA.tobsr(blocksize=(4,4)),B,('energy', {'SPD' : False, 'degree' : 2, 'maxiter' : 3}) ))
        cases.append((iA,iB,('energy', {'SPD' : False}) ))
        cases.append((iA.tobsr(blocksize=(4,4)),iB,('energy', {'SPD' : False}) ))

        A = gauge_laplacian(10, spacing=1.0, beta=0.21)
        B = ones((A.shape[0],1))
        cases.append((A,B,('energy', {'SPD' : True}) ))
        cases.append((A,iB,('energy', {'SPD' : True}) ))
        cases.append((A.tobsr(blocksize=(2,2)),iB,('energy', {'SPD' : True}) ))
        cases.append((A,B,('energy', {'SPD' : True, 'degree' : 2}) ))
        cases.append((A.tobsr(blocksize=(2,2)),B,('energy', {'SPD' : True, 'degree' : 2, 'maxiter' : 3}) ))
        cases.append((A,B,('energy', {'SPD' : False}) ))
        cases.append((A,B,('energy', {'SPD' : False, 'degree' : 2}) ))
        cases.append((A.tobsr(blocksize=(2,2)),B,('energy', {'SPD' : False, 'degree' : 2, 'maxiter' : 3}) ))

        A,B = linear_elasticity((10,10))
        cases.append((A,B,'energy'))
        cases.append((A,B,('energy', {'maxiter' : 3}) ))
        cases.append((A,B,('energy', {'SPD' : False}) ))
        cases.append((A,B,('energy', {'degree' : 2}) ))
        cases.append((A,B,('energy', {'SPD' : False, 'degree' : 2}) ))
        
        X = load_example('airfoil')
        A = X['A'].tocsr(); B = X['B']
        cases.append((A,B,'energy'))
        cases.append((A,B,('energy', {'maxiter' : 3}) ))
        cases.append((A,B,('energy', {'SPD' : False}) ))
        cases.append((A,B,('energy', {'degree' : 2}) ))
        cases.append((A,B,('energy', {'SPD' : False, 'degree' : 2}) ))
        

        for A,B,smooth in cases:
            ml = smoothed_aggregation_solver(A, B=B, max_coarse=1, max_levels=2, smooth=smooth )
            P = ml.levels[0].P
            B = ml.levels[0].B
            R = ml.levels[1].B
            assert_almost_equal(P*R, B)


        
    
#class TestSatisfyConstaints(TestCase):
#    def test_scalar(self):
#
#        U = bsr_matrix([[1,2],[2,1]], blocksize=(1,1))
#        Sparsity_Pattern = bsr_matrix([[1,1],[1,1]],blocksize=(1,1))
#        B = mat(array([[1],[1]]))
#        BtBinv = [ array([[0.5]]), array([[0.5]]) ]
#        colindices = [ array([0,1]), array([0,1]) ]
#
#        U = Satisfy_Constraints(U, Sparsity_Pattern, B, BtBinv, colindices)
#
#
#        assert_equal(U.todense(), array([[0,1],[1,0]]))
#        assert_almost_equal(U*B, 0*U*B)
#
#    def test_block(self):
#        SparsityPattern = array([[1,1,0,0],
#                                 [1,1,0,0],
#                                 [1,1,1,1],
#                                 [0,0,1,1],
#                                 [0,0,1,1]])
#        U = array([[1,2,0,0],
#                   [4,3,0,0],
#                   [5,6,8,7],
#                   [0,0,4,1],
#                   [0,0,2,3]])
#        
#        Sparsity_Pattern = bsr_matrix(SparsityPattern, blocksize=(1,2))
#        U = bsr_matrix(U, blocksize=(1,2))
#        B = array([[1,1],
#                   [1,2],
#                   [1,3],
#                   [1,4]])
#        colindices = [ array([0,1]), array([0,1]), array([0,1,2,3]), array([0,1]), array([0,1]) ]
#        
#        BtBinv = zeros((5,2,2)) 
#        for i in range(5):
#            colindx = colindices[i]
#            if len(colindx) > 0:
#                Bi = mat(B)[colindx,:]
#                BtBinv[i] = pinv2(Bi.T*Bi)
#
#        U = Satisfy_Constraints(U, Sparsity_Pattern, B, BtBinv, colindices)
#        assert_almost_equal(U*B, 0*U*B)
