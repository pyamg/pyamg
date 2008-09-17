from pyamg.testing import *

import numpy
from numpy import sqrt, ones, ravel, array
from scipy import rand
from scipy.sparse import csr_matrix, coo_matrix, spdiags
from scipy.special import round

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

    def test_ode_strength_of_connection(self):
        # Params:  A, B, epsilon=4.0, k=2, proj_type="l2"

        # Ensure that isotropic diffusion results in isotropic strength stencil
        for N in [3,5,7,10,11,19]:
            A = poisson( (N,), format='csr')
            B = ones((A.shape[0],1))
            Atilde = ode_strength_of_connection(A, B)
            assert_equal( Atilde.indptr, A.indptr )
            assert_equal( Atilde.indices, A.indices )

        for N in [3,5,7,10,11,19]:
            A = poisson( (N,N), format='csr') 
            B = ones((A.shape[0],1))
            Atilde = ode_strength_of_connection(A, B)
            assert_equal( Atilde.indptr, A.indptr )
            assert_equal( Atilde.indices, A.indices )
        
        # Ensure that anisotropic diffusion results an anisotropic strength stencil
        for N in [3,5,7,10,11,19]:
            A = spdiags([-ones(N*N), -0.1*ones(N*N), 2.2*ones(N*N),-0.1*ones(N*N),-ones(N*N)],[-N, -1, 0, 1, N], N*N, N*N, format='csr')
            AtildeExact = spdiags([-ones(N*N), 4*ones(N*N), -ones(N*N)],[-N, 0, N], N*N, N*N,format='csr')
            B = ones((A.shape[0],1))
            Atilde = ode_strength_of_connection(A, B)
            # Truncate the first and last rows
            Atilde = Atilde[1:-1]
            AtildeExact = AtildeExact[1:-1]
            assert_equal( Atilde.indptr, AtildeExact.indptr )
            assert_equal( Atilde.indices, AtildeExact.indices )
        
        for N in [3,5,7,10,11,19]:
            A = spdiags([-0.5*ones(N*N), -ones(N*N), 2.2*ones(N*N),-ones(N*N),-0.5*ones(N*N)],[-N, -1, 0, 1, N], N*N, N*N, format='csr')
            AtildeExact = spdiags([-ones(N*N), 4*ones(N*N), -ones(N*N)],[-1, 0, 1], N*N, N*N,format='csr')
            B = ones((A.shape[0],1))
            Atilde = ode_strength_of_connection(A, B, epsilon=2.0, k=4)
            # Truncate the first and last rows
            Atilde = Atilde[1:-1]
            AtildeExact = AtildeExact[1:-1]
            assert_equal( Atilde.indptr, AtildeExact.indptr )
            assert_equal( Atilde.indices, AtildeExact.indices )       

        # Ensure that isotropic elasticity results in an isotropic stencil (test bsr)
        for N in [3,5,7,10,11,19]:
            (A,B) = linear_elasticity( (N,N), format='bsr')
            Atilde = ode_strength_of_connection(A, B, proj_type="D_A",k=8,epsilon=32.0)
            AtildeExact =  csr_matrix((ones(A.indices.shape), A.indices, A.indptr), shape=(A.shape[0]/A.blocksize[0], A.shape[1]/A.blocksize[1]) )
            Atilde.sort_indices()
            AtildeExact.sort_indices()
            assert_equal( Atilde.indptr, AtildeExact.indptr )
            assert_equal( Atilde.indices, AtildeExact.indices )

        # Run a few "random" examples with pre-stored answers
        N = 5
        # CSR Test
        A1 = -1.0*array([ 0.21,  0.26,  0.15,  0.06,  0.14,  0.95,  0.8 ,  0.74,  0.57,
                    0.77,  0.77,  0.28,  0.17,  0.4 ,  0.47,  0.06,  0.91,  0.26,
                    0.81,  0.24,  0.21,  0.35,  0.63,  0.82,  0.39])
        A2 = -1.0*array([ 0.67,  0.25,  0.23,  0.42,  0.93,  0.04,  0.56,  0.97,  0.61,
                    0.91,  0.78,  0.48,  0.35,  0.64,  0.89,  0.05,  0.76,  0.35,
                    0.63  ,  0.04,  0.05,  0.66,  0.52,  0.33,  0.37])
        A3 = -1.0*array([ 0.21,  0.43,  0.25,  0.73,  0.27,  0.56,  0.8 ,  0.45,  0.37,
                    0.07,  0.65,  0.03,  0.85,  0.84,  0.48,  0.53,  0.73,  0.85,
                    0.43,  0.95,  0.55,  0.86,  0.04,  0.23,  0.54])
        A4 = -1.0*array([ 0.04,  0.51,  0.57,  0.52,  0.11,  0.8 ,  0.62,  0.74,  0.41,
                    0.82,  0.28,  0.93,  0.44,  0.26,  0.82,  0.55,  0.29,  0.16,
                    0.95,  0.71,  0.94,  0.73,  0.59,  0.51,  1.  ])
        A = spdiags([A1, A2, 4*ones(N*N),A3, A4],[-N, -1, 0, 1, N], N*N, N*N, format='csr')
        B = ones((A.shape[0],1))
        Atilde = ode_strength_of_connection(A, B, epsilon=8.0, proj_type="D_A")
        AtildeExact = csr_matrix((
        array([ 1.    ,  0.6645,  4.3104,  1.6258,  1.    ,  3.5164,  3.3427,
                3.3462,  1.    ,  3.7242,  6.2437,  0.4763,  1.    ,  1.5659,
                1.    ,  0.2778,  0.5596,  1.228 ,  1.    ,  0.3134,  1.0939,
                0.6228,  1.    ,  1.3183,  0.6228,  0.7066,  1.8063,  1.    ,
                0.3019,  0.7066,  1.792 ,  2.0939,  1.    ,  0.8766,  1.0083,
                0.6414,  1.    ,  0.4791,  0.748 ,  3.7392,  1.0415,  1.    ,
                0.7013,  0.7234,  0.414 ,  1.    ,  1.7397,  1.8743,  0.4878,
                1.    ,  2.6133,  3.9358,  0.5277,  1.    ,  1.0052,  2.2082,
                0.7521,  1.9932,  1.    ,  0.6143,  2.0569,  1.2691,  1.3548,
                1.    ,  3.6822,  0.8601,  1.    ,  0.7866,  0.4921,  0.465 ,
                1.    ,  2.5579,  0.4593,  2.224 ,  1.    ,  1.2005,  0.7115,
                0.8302,  0.3679,  1.    ,  0.0975,  1.    ,  0.6226,  0.3773,
                1.    ,  0.7947,  0.8238,  1.    ,  1.0694,  1.3114,  4.1253,
                1.    ,  2.5721,  0.1057,  1.    ]), 
        array([ 0,  1,  5,  0,  1,  2,  6,  1,  2,  3,  7,  2,  3,  4,  4,  5,  0,
                4,  5, 10,  1,  5,  6,  7, 11,  2,  6,  7,  8, 12,  3,  7,  8,  9,
               13,  4,  9, 10, 14,  5,  9, 10, 11, 15,  6, 11, 12,  7, 11, 12, 13,
                8, 12, 13, 14, 18,  9, 13, 14, 15, 19, 10, 14, 15, 11, 15, 16, 17,
               21, 16, 17, 18, 13, 17, 18, 19, 23, 14, 18, 19, 15, 20, 16, 20, 21,
               22, 17, 22, 23, 18, 22, 23, 24, 19, 24]), 
        array([ 0,  3,  7, 11, 14, 16, 20, 25, 30, 35, 39, 44, 47, 51, 56, 61, 64,
               69, 72, 77, 80, 82, 86, 89, 93, 95])
        ), shape=Atilde.shape)
        assert_equal( Atilde.indptr, AtildeExact.indptr )
        assert_equal( Atilde.indices, AtildeExact.indices )
        assert_almost_equal( Atilde.data, AtildeExact.data, decimal=2 )
        
        # BSR Test
        Absr = A.tobsr(blocksize=(5,5))
        Atilde = ode_strength_of_connection(Absr, B, epsilon=8.0, proj_type="D_A")
        AtildeExact = csr_matrix((
        array([ 1.    ,  3.3427,  0.5596,  1.    ,  0.3134,  0.414 ,  1.    ,
                0.7234,  0.4593,  1.    ,  0.4921,  0.0975,  1.    ]),
        array([0, 1, 0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4]),
        array([ 0,  2,  5,  8, 11, 13])
        ), shape=Atilde.shape)
        assert_equal( Atilde.indptr, AtildeExact.indptr )
        assert_equal( Atilde.indices, AtildeExact.indices )
        assert_almost_equal( Atilde.data, AtildeExact.data, decimal=2 )
        
        # Zero row CSR Test
        A.data[A.indptr[4]:A.indptr[5]] = 0.0
        A.eliminate_zeros()
        B = array([ravel(B), 
                   array([ 0.26,  0.37,  0.85,  0.48,  0.16,  0.49,  0.58,  0.7 ,  0.1 ,
                   0.78,  0.27,  0.28,  0.71,  0.18,  0.89,  0.65,  0.28,  0.89,
                   0.34,  0.04,  0.96,  0.25,  0.88,  0.58,  0.95]) ]).T
        Atilde = ode_strength_of_connection(A, B)
        AtildeExact = csr_matrix((
        ones((67,)),
        array([ 0,  1,  5,  1,  2,  2,  3,  2,  3,  0,  5, 10,  6, 11,  7,  8,  3,
                7,  8,  9,  9, 10, 14,  9, 10, 15,  6, 11, 12, 11, 12, 13, 14, 13,
               14, 10, 14, 15, 15, 16, 17, 21, 16, 17, 18, 13, 17, 18, 19, 23, 18,
               19, 15, 19, 20, 16, 20, 21, 22, 22, 23, 18, 22, 23, 24, 19, 24]),
        array([ 0,  3,  5,  7,  9,  9, 12, 14, 16, 20, 23, 26, 29, 31, 33, 35, 38,
               42, 45, 50, 52, 55, 59, 61, 65, 67])
        ), shape=Atilde.shape) 
        assert_equal( Atilde.indptr, AtildeExact.indptr )
        assert_equal( Atilde.indices, AtildeExact.indices )
        
        # Zero row BSR Test
        Absr = A.tobsr(blocksize=(5,5))
        Atilde = ode_strength_of_connection(Absr, B)
        AtildeExact = csr_matrix((
        array([  1.0,   1.0,   0.04256,  1.0,   0.07528,   1.0,
                 0.13055,   1e-8,   0.30166,   1.0,   1e-8,   1.0,   1.0]),
        #ones((13,)),
        array([0, 1, 0, 1, 2, 2, 3, 1, 2, 3, 4, 3, 4]),
        array([ 0,  2,  5,  8, 11, 13])
        ), shape=Atilde.shape)
        assert_equal( Atilde.indptr, AtildeExact.indptr )
        assert_equal( Atilde.indices, AtildeExact.indices )
        assert_almost_equal( Atilde.data, AtildeExact.data, decimal=2 )
        
        # Zero row and column CSR Test
        A = A.tocsc()
        A.data[A.indptr[4]:A.indptr[5]] = 0.0
        A.eliminate_zeros()
        A = A.tocsr()
        Atilde = ode_strength_of_connection(A, B)
        AtildeExact = csr_matrix((
        ones((67,)),
        array([ 0,  1,  5,  1,  2,  2,  3,  2,  3,  0,  5, 10,  6, 11,  7,  8,  3,
                7,  8,  9,  9, 10, 14,  9, 10, 15,  6, 11, 12, 11, 12, 13, 14, 13,
               14, 10, 14, 15, 15, 16, 17, 21, 16, 17, 18, 13, 17, 18, 19, 23, 18,
               19, 15, 19, 20, 16, 20, 21, 22, 22, 23, 18, 22, 23, 24, 19, 24]),
        array([ 0,  3,  5,  7,  9,  9, 12, 14, 16, 20, 23, 26, 29, 31, 33, 35, 38,
               42, 45, 50, 52, 55, 59, 61, 65, 67])
        ), shape=Atilde.shape)
        assert_equal( Atilde.indptr, AtildeExact.indptr )
        assert_equal( Atilde.indices, AtildeExact.indices )
        
        # Zero row and column BSR Test
        Absr = A.tobsr(blocksize=(5,5))
        Atilde = ode_strength_of_connection(Absr, B)
        AtildeExact = csr_matrix((
        ones((13,)),
        array([0, 1, 0, 1, 2, 2, 3, 1, 2, 3, 4, 3, 4]),
        array([ 0,  2,  5,  8, 11, 13])
        ), shape=Atilde.shape)
        assert_equal( Atilde.indptr, AtildeExact.indptr )
        assert_equal( Atilde.indices, AtildeExact.indices )
        
        
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


