from pyamg.testing import *

import numpy
from numpy import sqrt
from scipy import rand
from scipy.sparse import csr_matrix, coo_matrix

from pyamg.gallery import poisson, linear_elasticity, load_example
from pyamg.strength import *


class TestStrengthOfConnection(TestCase):
    def setUp(self):
        self.cases = []

        # random matrices
        numpy.random.seed(0)
        for N in [2,3,5]:
            self.cases.append( csr_matrix(rand(N,N)) )

        # poisson problems in 1D and 2D
        for N in [2,3,5,7,10,11,19]:
            self.cases.append( poisson( (N,), format='csr') )
        for N in [2,3,5,7,10,11]:
            self.cases.append( poisson( (N,N), format='csr') )

        for name in ['knot','airfoil','bar']:
            ex = load_example(name)
            self.cases.append( ex['A'].tocsr() )

    def test_classical_strength_of_connection(self):
        for A in self.cases:
            for theta in [ 0.0, 0.05, 0.25, 0.50, 0.90 ]:
                result   = classical_strength_of_connection(A, theta)
                expected = reference_classical_strength_of_connection(A, theta)
                
                assert_equal( result.nnz, expected.nnz )
                assert_equal( result.todense(), expected.todense() )

    def test_symmetric_strength_of_connection(self):
        for A in self.cases:
            for theta in [0.0, 0.1, 0.5, 1.0, 10.0]:
                expected = reference_symmetric_strength_of_connection(A, theta)
                result   = symmetric_strength_of_connection(A, theta)

                assert_equal( result.nnz,       expected.nnz)
                assert_equal( result.todense(), expected.todense())



################################################
##   reference implementations for unittests  ##
################################################
def reference_classical_strength_of_connection(A, theta):
    S = coo_matrix(A)
    
    # remove diagonals
    mask = S.row != S.col

    S.row  = S.row[mask]
    S.col  = S.col[mask]
    S.data = S.data[mask]
  
    min_offdiag    = numpy.empty(S.shape[0])
    min_offdiag[:] = numpy.finfo(S.data.dtype).max

    for i,v in zip(S.row,S.data):
        min_offdiag[i] = min(min_offdiag[i],v)

    # strong connections
    mask = S.data <= (theta * min_offdiag[S.row])
    
    S.row  = S.row[mask]
    S.col  = S.col[mask]
    S.data = S.data[mask]
    
    return S.tocsr()

def reference_symmetric_strength_of_connection(A, theta):
    #if theta == 0:
    #    return A
    
    D = abs(A.diagonal())

    S = coo_matrix(A)

    mask  = S.row != S.col
    mask &= abs(S.data) >= theta * sqrt(D[S.row] * D[S.col])

    S.row  = S.row[mask]
    S.col  = S.col[mask]
    S.data = S.data[mask]

    return S.tocsr()


