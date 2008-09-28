from pyamg.testing import *

from numpy import array, zeros, ones, mat
from scipy.linalg import pinv2
from scipy.sparse import bsr_matrix

from pyamg.aggregation.smooth import Satisfy_Constraints
from pyamg.gallery import poisson, linear_elasticity, load_example
from pyamg.aggregation import smoothed_aggregation_solver

class TestEnergyMin(TestCase):
    def test_range(self):
        """Check that P*R=B"""

        cases = []

        A = poisson((10,10), format='csr')
        B = ones((A.shape[0],1))
        cases.append((A,B))
        
        A,B = linear_elasticity((10,10))
        cases.append((A,B))
        
        X = load_example('airfoil')
        cases.append((X['A'].tocsr(),X['B']))

        opts = []
        opts.append( {} )
        opts.append( {'maxiter' : 3} )
        opts.append( {'SPD' : False} )
        opts.append( {'degree' : 2} )
        opts.append( {'SPD' : False, 'degree' : 2} )

        for A,B in cases:
            for kwargs in opts:
                ml = smoothed_aggregation_solver(A, B=B, max_coarse=1, max_levels=2, smooth=('energy',kwargs) )
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
