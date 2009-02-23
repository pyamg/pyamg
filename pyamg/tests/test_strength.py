from pyamg.testing import *

import numpy
from numpy import array, zeros, mat, eye, ones, setdiff1d, min, ravel, diff, mod, repeat, sqrt, finfo
from scipy import rand, real, imag, mat, zeros, sign, eye, arange
from scipy.sparse import csr_matrix, isspmatrix_csr, bsr_matrix, isspmatrix_bsr, spdiags, coo_matrix
import scipy.sparse
from scipy.special import round
from scipy.linalg import pinv2, pinv

from pyamg.gallery import poisson, linear_elasticity, load_example, stencil_grid
from pyamg.strength import *
from pyamg.multigridtools import incomplete_matmat
from pyamg.util.linalg import approximate_spectral_radius
from pyamg.util.utils import scale_rows


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

    def test_incomplete_matmat(self):
        # Test a critical helper routine for ode_strength_of_connection(...)
        # We test that (A*B).multiply(mask) = incomplete_matmat(A,B,mask)
        cases = []

        # 1x1 tests
        A = csr_matrix(mat([[1.1]]))
        B = csr_matrix(mat([[1.0]]))
        A2 = csr_matrix(mat([[0.]]))
        mask = csr_matrix(mat([[1.]]))
        cases.append( (A,A,mask) ) 
        cases.append( (A,B,mask) ) 
        cases.append( (A,A2,mask) ) 
        cases.append( (A2,A2,mask) ) 

        # 2x2 tests
        A = csr_matrix(mat([[1.,2.],[2.,4.]]))
        B = csr_matrix(mat([[1.3,2.],[2.8,4.]]))
        A2 = csr_matrix(mat([[1.3,0.],[0.,4.]]))
        B2 = csr_matrix(mat([[1.3,0.],[2.,4.]]))
        mask = csr_matrix( (ones(4),(array([0,0,1,1]),array([0,1,0,1]))), shape=(2,2) )
        cases.append( (A,A,mask) ) 
        cases.append( (A,B,mask) ) 
        cases.append( (A2,A2,mask) ) 
        cases.append( (A2,B2,mask) ) 

        mask = csr_matrix( (ones(3),(array([0,0,1]),array([0,1,1]))), shape=(2,2) )
        cases.append( (A,A,mask) ) 
        cases.append( (A,B,mask) ) 
        cases.append( (A2,A2,mask) ) 
        cases.append( (A2,B2,mask) ) 

        mask = csr_matrix( (ones(2),(array([0,1]),array([0,0]))), shape=(2,2) )
        cases.append( (A,A,mask) ) 
        cases.append( (A,B,mask) ) 
        cases.append( (A2,A2,mask) ) 
        cases.append( (A2,B2,mask) ) 

        # 5x5 tests
        A = mat([[  0. ,  16.9,   6.4,   0.0,   5.8],
                    [ 16.9,  13.8,   7.2,   0. ,   9.5],
                    [  6.4,   7.2,  12. ,   6.1,   5.9],
                    [  0.0,   0. ,   6.1,   0. ,   0. ],
                    [  5.8,   9.5,   5.9,   0. ,  13. ]])
        C = A.copy()
        C[1,0] = 3.1
        C[3,2] = 10.1
        A2 = A.copy()
        A2[1,:] = 0.0
        A3 = A2.copy()
        A3[:,1] = 0.0
        A = csr_matrix(A)
        A2 = csr_matrix(A2)
        A3 = csr_matrix(A3)
        C = csr_matrix(C)

        mask = A.copy()
        mask.data[:] = 1.0
        cases.append( (A,A,mask) )
        cases.append( (C,C,mask) )
        cases.append( (A2,A2,mask) )
        cases.append( (A3,A3,mask) )
        cases.append( (A,A2,mask) )
        cases.append( (A3,A,mask) )
        cases.append( (A,C,mask) )
        cases.append( (C,A,mask) )

        mask.data[1] = 0.0        
        mask.data[5] = 0.0        
        mask.data[9] = 0.0        
        mask.data[13] = 0.0        
        mask.eliminate_zeros()
        cases.append( (A,A,mask) )
        cases.append( (C,C,mask) )
        cases.append( (A2,A2,mask) )
        cases.append( (A3,A3,mask) )
        cases.append( (A,A2,mask) )
        cases.append( (A3,A,mask) )
        cases.append( (A,C,mask) )
        cases.append( (C,A,mask) )

        # Laplacian tests
        A = poisson((5,5),format='csr')
        B = A.copy()
        B.data[1] = 3.5
        B.data[11] = 11.6
        B.data[28] = -3.2
        C = csr_matrix(zeros(A.shape))
        mask=A.copy()
        mask.data[:]=1.0
        cases.append( (A,A,mask) )
        cases.append( (A,B,mask) )
        cases.append( (B,A,mask) )
        cases.append( (C,A,mask) )
        cases.append( (A,C,mask) )
        cases.append( (C,C,mask) )

        # Imaginary tests
        A = mat([[  0.0 +0.j ,   0.0+16.9j,   6.4 +1.2j,   0.0 +0.j ,   0.0 +0.j ],
                 [ 16.9 +0.j ,  13.8 +0.j ,   7.2 +0.j ,   0.0 +0.j ,   0.0 +9.5j],
                 [  0.0 +6.4j,   7.2 -8.1j,  12.0 +0.j ,   6.1 +0.j ,   5.9 +0.j ],
                 [  0.0 +0.j ,   0.0 +0.j ,   6.1 +0.j ,   0.0 +0.j ,   0.0 +0.j ],
                 [  5.8 +0.j ,  -4.0 +9.5j,  -3.2 -5.9j,   0.0 +0.j ,  13.0 +0.j ]])
        C = A.copy()
        C[1,0] = 3.1j - 1.3
        C[3,2] = -10.1j + 9.7
        A = csr_matrix(A)
        C = csr_matrix(C)

        mask = A.copy()
        mask.data[:] = 1.0
        cases.append( (A,A,mask) )
        cases.append( (C,C,mask) )
        cases.append( (A,C,mask) )
        cases.append( (C,A,mask) )


        for case in cases:
            A = case[0].tocsr()
            B = case[1].tocsc()
            mask = case[2].tocsr()
            A.sort_indices()
            B.sort_indices()
            mask.sort_indices()
            result = mask.copy()
            incomplete_matmat(A.indptr,A.indices,A.data, B.indptr,B.indices,B.data, 
                              result.indptr,result.indices,result.data, A.shape[0])
            result.eliminate_zeros()
            exact = (A*B).multiply(mask)
            exact.sort_indices()
            exact.eliminate_zeros()
            assert_array_almost_equal(exact.data, result.data)
            assert_array_equal(exact.indptr, result.indptr)
            assert_array_equal(exact.indices, result.indices)


    def test_ode_strength_of_connection(self):
        # Params:  A, B, epsilon=4.0, k=2, proj_type="l2"

        # Ensure that isotropic diffusion results in isotropic strength stencil
        for N in [3,5,7,10,11,19]:
            A = poisson( (N,), format='csr')
            B = ones((A.shape[0],1))
            result = ode_strength_of_connection(A, B, epsilon=4.0, k=2, proj_type="l2")
            expected = reference_ode_strength_of_connection(A, B, epsilon=4.0, k=2, proj_type="12")
            assert_array_almost_equal( result.todense(), expected.todense() )

        for N in [3,5,7,10,11,19]:
            A = poisson( (N,N), format='csr') 
            B = ones((A.shape[0],1))
            result = ode_strength_of_connection(A, B, epsilon=4.0, k=2, proj_type="l2")
            expected = reference_ode_strength_of_connection(A, B, epsilon=4.0, k=2, proj_type="12")
            assert_array_almost_equal( result.todense(), expected.todense() )
        
        # Ensure that anisotropic diffusion results an anisotropic strength stencil
        for N in [3,5,7,10,11,19]:
            A = spdiags([-ones(N*N), -0.1*ones(N*N), 2.2*ones(N*N),-0.1*ones(N*N),-ones(N*N)],[-N, -1, 0, 1, N], N*N, N*N, format='csr')
            B = ones((A.shape[0],1))
            result = ode_strength_of_connection(A, B, epsilon=4.0, k=2, proj_type="l2")
            expected = reference_ode_strength_of_connection(A, B, epsilon=4.0, k=2, proj_type="12")
            assert_array_almost_equal( result.todense(), expected.todense() )

        for N in [3,5,7,10,11,19]:
            A = spdiags([-0.5*ones(N*N), -ones(N*N), 2.2*ones(N*N),-ones(N*N),-0.5*ones(N*N)],[-N, -1, 0, 1, N], N*N, N*N, format='csr')
            B = ones((A.shape[0],1))
            result = ode_strength_of_connection(A, B, epsilon=4.0, k=2, proj_type="l2")
            expected = reference_ode_strength_of_connection(A, B, epsilon=4.0, k=2, proj_type="12")
            assert_array_almost_equal( result.todense(), expected.todense() )
            
        # Ensure that isotropic elasticity results in an isotropic stencil (test bsr)
        for N in [3,5,7,10]:
            (A,B) = linear_elasticity( (N,N), format='bsr')
            result = ode_strength_of_connection(A, B, epsilon=32.0, k=8, proj_type="D_A")
            expected = reference_ode_strength_of_connection(A, B, epsilon=32.0, k=8, proj_type="D_A")
            assert_array_almost_equal( result.todense(), expected.todense() )


        # Run a few "random" examples
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
        result = ode_strength_of_connection(A+A.T, B, epsilon=8.0, proj_type="D_A")
        expected = reference_ode_strength_of_connection(A+A.T, B, epsilon=8.0, proj_type="D_A")
        assert_array_almost_equal( result.todense(), expected.todense() )

        # BSR Test
        Absr = A.tobsr(blocksize=(5,5))
        result = ode_strength_of_connection(Absr, B, epsilon=8.0, k=2, proj_type="D_A")
        expected = reference_ode_strength_of_connection(Absr, B, epsilon=8.0, k=2, proj_type="D_A")
        assert_array_almost_equal( result.todense(), expected.todense())
        
        # Different B CSR Test
        B = array([ravel(B), 
                   array([ 0.26,  0.37,  0.85,  0.48,  0.16,  0.49,  0.58,  0.7 ,  0.1 ,
                   0.78,  0.27,  0.28,  0.71,  0.18,  0.89,  0.65,  0.28,  0.89,
                   0.34,  0.04,  0.96,  0.25,  0.88,  0.58,  0.95]) ]).T
        result = ode_strength_of_connection(A, B, epsilon=4.0, k=2, proj_type="l2")
        expected = reference_ode_strength_of_connection(A, B, epsilon=4.0, k=2, proj_type="l2")
        assert_array_almost_equal( result.todense(), expected.todense() )
        
        # BSR Test
        Absr = A.tobsr(blocksize=(5,5))
        result = ode_strength_of_connection(Absr, B, epsilon=4.0, k=2, proj_type="l2")
        expected = reference_ode_strength_of_connection(Absr, B, epsilon=4.0, k=2, proj_type="l2")
        assert_array_almost_equal( result.todense(), expected.todense() )
        
        # Zero row and column CSR Test
        A.data[A.indptr[4]:A.indptr[5]] = 0.0
        A = A.tocsc()
        A.data[A.indptr[4]:A.indptr[5]] = 0.0
        A.eliminate_zeros()
        A = A.tocsr()
        A.sort_indices()
        result = ode_strength_of_connection(A, B, epsilon=4.0, k=2, proj_type="l2")
        expected = reference_ode_strength_of_connection(A, B, epsilon=4.0, k=2, proj_type="l2")
        assert_array_almost_equal( result.todense(), expected.todense() )

        # Zero row and column BSR Test
        Absr = A.tobsr(blocksize=(5,5))
        result = ode_strength_of_connection(Absr, B, epsilon=4.0, k=2, proj_type="l2")
        expected = reference_ode_strength_of_connection(Absr, B, epsilon=4.0, k=2, proj_type="l2")
        assert_array_almost_equal( result.todense(), expected.todense() )

        # Test Scale Invariance
        A = poisson( (5,5), format='csr') 
        B = arange(1,A.shape[0]+1,dtype=float).reshape(-1,1)
        result_unscaled = ode_strength_of_connection(A, B, epsilon=4.0, k=2, proj_type="D_A")
        # create scaled A
        D = spdiags([arange(A.shape[0],2*A.shape[0],dtype=float)], [0], A.shape[0], A.shape[0], format = 'csr')
        Dinv = spdiags([1.0/arange(A.shape[0],2*A.shape[0],dtype=float)], [0], A.shape[0], A.shape[0], format = 'csr')
        Ascaled = D*A*D
        Bscaled = Dinv*B
        result_scaled = ode_strength_of_connection(Ascaled, Bscaled, epsilon=4.0, k=2, proj_type="D_A")
        assert_array_almost_equal( result_scaled.todense(), result_unscaled.todense(), decimal=2 )

        # Test that the l2 and D_A are the same for the 1 near nullspace candidate case
        resultDA = ode_strength_of_connection(Ascaled, Bscaled, epsilon=4.0, k=2, proj_type="D_A")
        resultl2 = ode_strength_of_connection(Ascaled, Bscaled, epsilon=4.0, k=2, proj_type="l2")
        assert_array_almost_equal( resultDA.todense(), resultl2.todense() )

        # Multiple Near Nullspace Candidates
        (A,B) = linear_elasticity( (5,5), format='bsr')
        result_unscaled = ode_strength_of_connection(A, B, epsilon=4.0, k=2, proj_type="D_A")
        # create scaled A
        D = spdiags([arange(A.shape[0],2*A.shape[0],dtype=float)], [0], A.shape[0], A.shape[0], format = 'csr')
        Dinv = spdiags([1.0/arange(A.shape[0],2*A.shape[0],dtype=float)], [0], A.shape[0], A.shape[0], format = 'csr')
        Ascaled = (D*A*D).tobsr(blocksize=(2,2))
        Bscaled = Dinv*B
        result_scaled = ode_strength_of_connection(Ascaled, Bscaled, epsilon=4.0, k=2, proj_type="D_A")
        assert_array_almost_equal( result_scaled.todense(), result_unscaled.todense(), decimal=2 )


# Define Complex tests
class TestComplexStrengthOfConnection(TestCase):
    def setUp(self):
        self.cases = []

        # random matrices
        numpy.random.seed(0)
        for N in [2,3,5]:
            self.cases.append( csr_matrix(rand(N,N)) + csr_matrix(1.0j*rand(N,N)))

        # poisson problems in 1D and 2D
        for N in [2,3,5,7,10,11,19]:
            A = poisson( (N,), format='csr'); A.data = A.data + 1.0j*A.data;
            self.cases.append(A)
        for N in [2,3,5,7,10,11]:
            A = poisson( (N,N), format='csr'); A.data = A.data + 1.0j*rand(A.data.shape[0],);
            self.cases.append(A)

        for name in ['knot','airfoil','bar']:
            ex = load_example(name)
            A = ex['A'].tocsr(); A.data = A.data + 0.5j*rand(A.data.shape[0],);
            self.cases.append(A)

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
        # Single near nullspace candidate
        stencil = [[0.0, -1.0, 0.0],[-0.001, 2.002, -0.001],[0.0, -1.0, 0.0]]
        A = 1.0j*stencil_grid(stencil, (4,4), format='csr')
        B = 1.0j*ones((A.shape[0],1))
        result = ode_strength_of_connection(A, B, epsilon=4.0, k=2, proj_type="l2")
        expected = reference_ode_strength_of_connection(A, B, epsilon=4.0, k=2, proj_type="12")
        assert_array_almost_equal( result.todense(), expected.todense() )
        # Tweak B
        B[0] = 1.2 - 12.0j
        B[11] = -14.2
        result = ode_strength_of_connection(A, B, epsilon=4.0, k=2, proj_type="l2")
        expected = reference_ode_strength_of_connection(A, B, epsilon=4.0, k=2, proj_type="12")
        assert_array_almost_equal( result.todense(), expected.todense() )

        # Multiple near nullspace candidate
        B = 1.0j*ones((A.shape[0],2))
        B[0:-1:2,0] = 0.0
        B[1:-1:2,1] = 0.0
        B[-1,0] = 0.0
        result = ode_strength_of_connection(A, B, epsilon=4.0, k=2, proj_type="l2")
        expected = reference_ode_strength_of_connection(A, B, epsilon=4.0, k=2, proj_type="12")
        assert_array_almost_equal( result.todense(), expected.todense() )
        # Different B
        B[11,1] = -14.2
        B[0,0] = 1.2 - 12.0j
        result = ode_strength_of_connection(A, B, epsilon=4.0, k=2, proj_type="l2")
        expected = reference_ode_strength_of_connection(A, B, epsilon=4.0, k=2, proj_type="12")
        assert_array_almost_equal( result.todense(), expected.todense() )
        
        B = array(arange(1,2*A.shape[0]+1).reshape(-1,2),dtype=complex)
        result = ode_strength_of_connection(A, B, epsilon=4.0, k=2, proj_type="l2")
        expected = reference_ode_strength_of_connection(A, B, epsilon=4.0, k=2, proj_type="12")
        assert_array_almost_equal( result.todense(), expected.todense() )

        # Multiple near nullspace candidate, BSR Test
        A = A.tobsr(blocksize=(2,2))
        result = ode_strength_of_connection(A, B, epsilon=4.0, k=2, proj_type="l2")
        expected = reference_ode_strength_of_connection(A, B, epsilon=4.0, k=2, proj_type="12")
        assert_array_almost_equal( result.todense(), expected.todense() )
        # Tweak B
        B[0,1] = -9.2 + 2.3j
        B[11,0] = 14.2
        result = ode_strength_of_connection(A, B, epsilon=4.0, k=2, proj_type="l2")
        expected = reference_ode_strength_of_connection(A, B, epsilon=4.0, k=2, proj_type="12")
        assert_array_almost_equal( result.todense(), expected.todense(), decimal=2 )

        # Test Scale Invariance
        A = 1.0j*poisson( (5,5), format='csr') 
        B = 1.0j*arange(1,A.shape[0]+1,dtype=float).reshape(-1,1)
        result_unscaled = ode_strength_of_connection(A, B, epsilon=4.0, k=2, proj_type="D_A")
        # create scaled A
        D = spdiags([arange(A.shape[0],2*A.shape[0],dtype=float)], [0], A.shape[0], A.shape[0], format = 'csr')
        Dinv = spdiags([1.0/arange(A.shape[0],2*A.shape[0],dtype=float)], [0], A.shape[0], A.shape[0], format = 'csr')
        Ascaled = D*A*D
        Bscaled = Dinv*B
        result_scaled = ode_strength_of_connection(Ascaled, Bscaled, epsilon=4.0, k=2, proj_type="D_A")
        assert_array_almost_equal( result_scaled.todense(), result_unscaled.todense(), decimal=2 )
        
        # Test that the l2 and D_A are the same for the 1 near nullspace candidate case
        resultDA = ode_strength_of_connection(Ascaled, Bscaled, epsilon=4.0, k=2, proj_type="D_A")
        resultl2 = ode_strength_of_connection(Ascaled, Bscaled, epsilon=4.0, k=2, proj_type="l2")
        assert_array_almost_equal( resultDA.todense(), resultl2.todense() )

        # Multiple Near Nullspace Candidates
        (A,B) = linear_elasticity( (5,5), format='bsr')
        A = 1.0j*A
        B = 1.0j*B
        result_unscaled = ode_strength_of_connection(A, B, epsilon=4.0, k=2, proj_type="D_A")
        # create scaled A
        D = spdiags([arange(A.shape[0],2*A.shape[0],dtype=float)], [0], A.shape[0], A.shape[0], format = 'csr')
        Dinv = spdiags([1.0/arange(A.shape[0],2*A.shape[0],dtype=float)], [0], A.shape[0], A.shape[0], format = 'csr')
        Ascaled = (D*A*D).tobsr(blocksize=(2,2))
        Bscaled = Dinv*B
        result_scaled = ode_strength_of_connection(Ascaled, Bscaled, epsilon=4.0, k=2, proj_type="D_A")
        assert_array_almost_equal( result_scaled.todense(), result_unscaled.todense(), decimal=2 )


################################################
##   reference implementations for unittests  ##
################################################
def reference_classical_strength_of_connection(A, theta):
    # This complex extension of the classic Ruge-Stuben 
    # strength-of-connection has some theoretical justification in
    # "AMG Solvers for Complex-Valued Matrices", Scott MacClachlan, 
    # Cornelis Oosterlee

    # Connection is strong if, 
    #   | a_ij| >= theta * max_{k != i} |a_ik|
    S = coo_matrix(A)
    
    # remove diagonals
    mask = S.row != S.col
    S.row  = S.row[mask]
    S.col  = S.col[mask]
    S.data = S.data[mask]
    max_offdiag    = numpy.empty(S.shape[0])
    max_offdiag[:] = numpy.finfo(S.data.dtype).min

    # Note abs(.) takes the complex modulus
    for i,v in zip(S.row,S.data):
        max_offdiag[i] = max(max_offdiag[i], abs(v))

    # strong connections
    mask = abs(S.data) >= (theta * max_offdiag[S.row])
    S.row  = S.row[mask]
    S.col  = S.col[mask]
    S.data = S.data[mask]

    return S.tocsr()
    

def reference_symmetric_strength_of_connection(A, theta):
    # This is just a direct complex extension of the classic 
    # SA strength-of-connection measure.  The extension continues 
    # to compare magnitudes. This should reduce to the classic 
    # measure if A is all real.

    #if theta == 0:
    #    return A
    
    D = abs(A.diagonal())

    S = coo_matrix(A)

    mask  = S.row != S.col
    DD = array(D[S.row] * D[S.col]).reshape(-1,)
    # Note that abs takes the complex modulus element-wise
    # Note that using the square of the measure is the technique used 
    # in the C++ routine, so we use it here.  Doing otherwise causes errors. 
    mask &= (  (real(S.data)**2 + imag(S.data)**2) >= theta*theta*DD )

    S.row  = S.row[mask]
    S.col  = S.col[mask]
    S.data = S.data[mask]

    return S.tocsr()



def reference_ode_strength_of_connection(A, B, epsilon=4.0, k=2, proj_type="l2"):
    """
    All python reference implementation for ODE Strength of Connection
    If doing imaginary test, both A and B should be imaginary type upon entry
    """
   
    #number of PDEs per point is defined implicitly by block size
    csrflag = isspmatrix_csr(A)
    if csrflag:
        numPDEs = 1
    else:
        numPDEs = A.blocksize[0]
        A = A.tocsr()
    
    # Preliminaries
    near_zero = finfo(float).eps
    sqrt_near_zero = sqrt(sqrt(near_zero))
    Bmat = mat(B)
    A.eliminate_zeros()
    A.sort_indices()
    dimen = A.shape[1]
    NullDim = Bmat.shape[1]

    #Get spectral radius of Dinv*A, this is the time step size for the ODE
    D = A.diagonal();
    Dinv = 1.0 / D
    Dinv[D == 0] = 1.0
    Dinv_A  = scale_rows(A, Dinv, copy=True)
    rho_DinvA = approximate_spectral_radius(Dinv_A)
     
    # Calculate (Atilde^k) naively  
    S = (scipy.sparse.eye(dimen,dimen,format="csr") - (1.0/rho_DinvA)*Dinv_A)
    Atilde = scipy.sparse.eye(dimen, dimen, format="csr")
    for i in range(k):
        Atilde = S*Atilde
    
    # Strength Info should be row-based, so transpose Atilde
    Atilde = Atilde.T.tocsr()

    #====================================================================
    #Construct and apply a sparsity mask for Atilde that restricts Atilde^T to the nonzero pattern
    #  of A, with the added constraint that row i of Atilde^T retains only the nonzeros that are also
    #  in the same PDE as i.

    mask = A.copy()

    #Only consider strength at dofs from your PDE.  Use mask to enforce this by zeroing out
    #   all entries in Atilde that aren't from your PDE.
    if numPDEs > 1:
        row_length = diff(mask.indptr)
        my_pde = mod(range(dimen), numPDEs)
        my_pde = repeat(my_pde, row_length)
        mask.data[ mod(mask.indices, numPDEs) != my_pde ] = 0.0
        del row_length, my_pde
        mask.eliminate_zeros()

    #Apply mask to Atilde, zeros in mask have already been eliminated at start of routine.
    mask.data[:] = 1.0
    Atilde = Atilde.multiply(mask)
    Atilde.eliminate_zeros()
    Atilde.sort_indices()
    del mask

    #====================================================================
    # Calculate strength based on constrained min problem of
    LHS = mat(zeros((NullDim+1, NullDim+1)), dtype=A.dtype)
    RHS = mat(zeros((NullDim+1, 1)), dtype=A.dtype)
   
    for i in range(dimen):
       
        #Get rowptrs and col indices from Atilde
        rowstart = Atilde.indptr[i]
        rowend = Atilde.indptr[i+1]
        length = rowend - rowstart
        colindx = Atilde.indices[rowstart:rowend]
       
        # Local diagonal of A is used for scale invariant min problem
        D_A = mat(eye(length))
        if proj_type == "D_A":
            for j in range(length):
                D_A[j,j] = D[colindx[j]] 

        #Find row i's position in colindx, matrix must have sorted column indices.
        iInRow = colindx.searchsorted(i)
   
        if length <= NullDim:
            #Do nothing, because the number of nullspace vectors will  
            #be able to perfectly approximate this row of Atilde.
            Atilde.data[rowstart:rowend] = 1.0
        else:
            #Grab out what we want from Atilde and B.  Put into zi, Bi
            zi = mat(Atilde.data[rowstart:rowend]).T
           
            Bi = Bmat[colindx,:]
   
            #Construct constrained min problem
            LHS[0:NullDim, 0:NullDim] = 2.0*Bi.H*D_A*Bi
            LHS[0:NullDim, NullDim] = D_A[iInRow,iInRow]*Bi[iInRow,:].H  
            LHS[NullDim, 0:NullDim] = Bi[iInRow,:]
            RHS[0:NullDim,0] = 2.0*Bi.H*D_A*zi
            RHS[NullDim,0] = zi[iInRow,0]

            #Calc Soln to Min Problem
            x = mat(pinv(LHS))*RHS
            
            #Calc best constrained approximation to zi with span(Bi).  
            zihat = Bi*x[:-1]
            
            #if angle in the complex plane between individual entries is 
            #   greater than 90 degrees, then weak.  We can just look at the
            #   dot product to determine if angle is greater than 90 degrees.
            angle = real(ravel(zihat))*real(ravel(zi)) + imag(ravel(zihat))*imag(ravel(zi))
            angle[angle < 0.0] = True
            angle[angle >= 0.0] = False
            angle = array(angle, dtype=bool)

            #Calculate approximation ratio
            zi = zihat/zi
           
            # If the ratio is small, then weak
            zi[abs(zi) <= 1e-4] = 1e100 

            # If angle is greater than 90 degrees, then weak
            zi[angle] = 1e100

            #Calculate Relative Approximation Error
            zi = abs(1.0 - zi)
            
            # important to make "perfect" connections explicitly nonzero
            zi[zi < sqrt_near_zero] = 1e-4                 

            #Calculate and applydrop-tol.  Ignore diagonal by making it very large
            zi[iInRow] = 1e5
            drop_tol = min(zi)*epsilon
            zi[zi > drop_tol] = 0.0
            Atilde.data[rowstart:rowend] = ravel(zi)

    #===================================================================
    # Set diagonal to 1.0, as each point is strongly connected to itself.
    I = scipy.sparse.eye(dimen, dimen, format="csr")
    I.data -= Atilde.diagonal()
    Atilde = Atilde + I

    # Clean up, and return Atilde
    Atilde.eliminate_zeros()
    Atilde.data = array(Atilde.data, dtype=float)

    # If converted BSR to CSR we return amalgamated matrix with the minimum nonzero for each block 
    # making up the nonzeros of Atilde
    if not csrflag:
        Atilde = Atilde.tobsr(blocksize=(numPDEs, numPDEs))
        
        #Atilde = csr_matrix((data, row, col), shape=(*,*))
        Atilde = csr_matrix((array([ Atilde.data[i,:,:][Atilde.data[i,:,:].nonzero()].min() for i in range(Atilde.indices.shape[0]) ]), \
                             Atilde.indices, Atilde.indptr), shape=(Atilde.shape[0]/numPDEs, Atilde.shape[1]/numPDEs) )

    return Atilde


