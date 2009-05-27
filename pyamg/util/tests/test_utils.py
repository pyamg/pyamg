from pyamg.testing import *

from numpy import matrix, array, diag, zeros, sqrt, abs, ravel
from scipy import rand, linalg, real, imag, mat, diag, isscalar, ones
from scipy.sparse import csr_matrix, isspmatrix, bsr_matrix

from pyamg.util.utils import *
from pyamg.util.utils import symmetric_rescaling

class TestUtils(TestCase):
    def test_diag_sparse(self):
        #check sparse -> array
        A = matrix([[-4]])
        assert_equal(diag_sparse(csr_matrix(A)),[-4])

        A = matrix([[1,0,-5],[-2,5,0]])
        assert_equal(diag_sparse(csr_matrix(A)),[1,5])

        A = matrix([[0,1],[0,-5]])
        assert_equal(diag_sparse(csr_matrix(A)),[0,-5])

        A = matrix([[1.3,-4.7,0],[-2.23,5.5,0],[9,0,-2]])
        assert_equal(diag_sparse(csr_matrix(A)),[1.3,5.5,-2])

        #check array -> sparse
        A = matrix([[-4]])
        assert_equal(diag_sparse(array([-4])).todense(),csr_matrix(A).todense())

        A = matrix([[1,0],[0,5]])
        assert_equal(diag_sparse(array([1,5])).todense(),csr_matrix(A).todense())

        A = matrix([[0,0],[0,-5]])
        assert_equal(diag_sparse(array([0,-5])).todense(),csr_matrix(A).todense())

        A = matrix([[1.3,0,0],[0,5.5,0],[0,0,-2]])
        assert_equal(diag_sparse(array([1.3,5.5,-2])).todense(),csr_matrix(A).todense())


    def test_symmetric_rescaling(self):
        cases = []
        cases.append( diag_sparse(array([1,2,3,4])) )
        cases.append( diag_sparse(array([1,0,3,4])) )

        A = array([ [ 5.5,  3.5,  4.8],
                    [ 2. ,  9.9,  0.5],
                    [ 6.5,  2.6,  5.7]])
        A = csr_matrix( A )
        cases.append(A)
        P = diag_sparse([1,0,1])
        cases.append( P*A*P )
        P = diag_sparse([0,1,0])
        cases.append( P*A*P )
        P = diag_sparse([1,-1,1])
        cases.append( P*A*P )

        for A in cases:
            D_sqrt,D_sqrt_inv,DAD = symmetric_rescaling(A)

            assert_almost_equal( diag_sparse(A) > 0, diag_sparse(DAD) )
            assert_almost_equal( diag_sparse(DAD), D_sqrt*D_sqrt_inv )

            D_sqrt,D_sqrt_inv = diag_sparse(D_sqrt),diag_sparse(D_sqrt_inv)
            assert_almost_equal((D_sqrt_inv*A*D_sqrt_inv).todense(), DAD.todense())


    def test_profile_solver(self):
        from scipy.sparse.linalg import cg
        from pyamg.gallery import poisson
        from pyamg.aggregation import smoothed_aggregation_solver

        A = poisson((100,100), format='csr')
        ml = smoothed_aggregation_solver(A)

        opts = []
        opts.append( {} )
        opts.append( {'accel' : cg } )
        opts.append( {'accel' : cg, 'tol' : 1e-10 } )

        for kwargs in opts:
            residuals = profile_solver(ml, **kwargs)

    def test_get_block_diag(self):
        from scipy import arange, ravel, array
        from scipy.sparse import csr_matrix
        A = csr_matrix(arange(1,37, dtype=float).reshape(6,6))
        block_diag = get_block_diag(A, blocksize=1, inv_flag=False)
        assert_array_almost_equal(ravel(block_diag), A.diagonal())

        block_diag = get_block_diag(A, blocksize=2, inv_flag=False)
        answer = array([[[  1.,   2.],
                         [  7.,   8.]],
                        [[ 15.,  16.],
                         [ 21.,  22.]],
                        [[ 29.,  30.],
                         [ 35.,  36.]]])
        assert_array_almost_equal(ravel(block_diag), ravel(answer))

        block_diag_inv = get_block_diag(A, blocksize=2, inv_flag=True)
        answer = array([[[-1.33333333,  0.33333333],
                         [ 1.16666667, -0.16666667]],
                        [[-3.66666667,  2.66666667],
                         [ 3.5       , -2.5       ]],
                        [[-6.        ,  5.        ],
                         [ 5.83333333, -4.83333333]]])
        assert_array_almost_equal(ravel(block_diag_inv), ravel(answer), decimal=3)

        # try with singular (1,1) block, a zero (2,2) block and a zero (0,2) block
        A = bsr_matrix(array([ 1.,  2.,  3.,  4.,  0.,  0.,
                               5.,  6.,  7.,  8.,  0.,  0.,
                               9., 10., 11., 11., 12., 13.,
                              14., 15., 16., 16., 18., 19.,
                              20., 21., 22., 23.,  0.,  0.,
                              26., 27., 28., 29.,  0.,  0.,]).reshape(6,6), blocksize=(3,3))
        block_diag_inv = get_block_diag(A, blocksize=2, inv_flag=True)
        answer = array([[[-1.5       ,  0.5       ],
                         [ 1.25      , -0.25      ]],
                        [[ 0.01458886,  0.02122016],
                         [ 0.01458886,  0.02122016]],
                        [[ 0.        ,  0.        ],
                         [ 0.        ,  0.        ]]])
        assert_array_almost_equal(ravel(block_diag_inv), ravel(answer), decimal=3)

        # try with different types of zero blocks
        A = bsr_matrix(array([ 0.,  0.,  3.,  4.,  0.,  0.,
                               0.,  0.,  7.,  8.,  0.,  0.,
                               0.,  0.,  0.,  0.,  0.,  0.,
                               0.,  0.,  0.,  0.,  0.,  0.,
                               0.,  0.,  0.,  0., 22., 23.,
                               0.,  0.,  0.,  0., 28., 29.,]).reshape(6,6), blocksize=(2,2))
        block_diag_inv = get_block_diag(A, blocksize=2, inv_flag=False)
        answer = array([[[ 0.        ,  0.        ],
                         [ 0.        ,  0.        ]],
                        [[ 0.        ,  0.        ],
                         [ 0.        ,  0.        ]],
                        [[22.        , 23.        ],
                         [28.        , 29.        ]]])
        assert_array_almost_equal(ravel(block_diag_inv), ravel(answer), decimal=3)


class TestComplexUtils(TestCase):
    def test_diag_sparse(self):
        #check sparse -> array
        A = matrix([[-4-4.0j]])
        assert_equal(diag_sparse(csr_matrix(A)),[-4-4.0j])

        A = matrix([[1,0,-5],[-2,5-2.0j,0]])
        assert_equal(diag_sparse(csr_matrix(A)),[1,5-2.0j])

        #check array -> sparse
        A = matrix([[-4+1.0j]])
        assert_equal(diag_sparse(array([-4+1.0j])).todense(),csr_matrix(A).todense())

        A = matrix([[1,0],[0,5-2.0j]])
        assert_equal(diag_sparse(array([1,5-2.0j])).todense(),csr_matrix(A).todense())

    def test_symmetric_rescaling(self):
        cases = []
        A = array([ [ 5.5+1.0j,  3.5,    4.8   ],
                    [ 2. ,       9.9,  0.5-2.0j],
                    [ 6.5,       2.6,  5.7+1.0j]])
        A = csr_matrix( A )
        cases.append(A)
        P = diag_sparse([1,0,1.0j])
        cases.append( P*A*P )
        P = diag_sparse([0,1+1.0j,0])
        cases.append( P*A*P )

        for A in cases:
            D_sqrt,D_sqrt_inv,DAD = symmetric_rescaling(A)
            assert_almost_equal( diag_sparse(A) != 0, real(diag_sparse(DAD)) )
            assert_almost_equal( diag_sparse(DAD), D_sqrt*D_sqrt_inv )

            D_sqrt,D_sqrt_inv = diag_sparse(D_sqrt),diag_sparse(D_sqrt_inv)
            assert_almost_equal((D_sqrt_inv*A*D_sqrt_inv).todense(), DAD.todense())

    def test_get_diagonal(self):
        cases = []
        for i in range(1,6):
            A = rand(i,i)
            Ai = A + 1.0j*rand(i,i)
            cases.append(csr_matrix(A)) 
            cases.append(csr_matrix(Ai)) 


        for A in cases:
            D_A       = get_diagonal(A, norm_eq=False, inv=False)
            D_A_inv   = get_diagonal(A, norm_eq=False, inv=True)
            D_AA      = get_diagonal(A, norm_eq=1, inv=False)
            D_AA_inv  = get_diagonal(A, norm_eq=1, inv=True)
            D_AA2     = get_diagonal(A, norm_eq=2, inv=False)
            D_AA_inv2 = get_diagonal(A, norm_eq=2, inv=True)
            
            D = diag(A.todense())
            assert_almost_equal(D, D_A)
            D = 1.0/D
            assert_almost_equal(D, D_A_inv)
            
            D = diag((A.H*A).todense())
            assert_almost_equal(D, D_AA)
            D = 1.0/D
            assert_almost_equal(D, D_AA_inv)
            
            D = diag((A*A.H).todense())
            assert_almost_equal(D, D_AA2)
            D = 1.0/D
            assert_almost_equal(D, D_AA_inv2)


    def test_profile_solver(self):
        from scipy.sparse.linalg import cg
        from pyamg.gallery import poisson
        from pyamg.aggregation import smoothed_aggregation_solver

        A = poisson((100,100), format='csr')
        A.data = A.data + 1e-5*rand(A.nnz)
        ml = smoothed_aggregation_solver(A)

        opts = []
        opts.append( {} )
        opts.append( {'accel' : cg } )
        opts.append( {'accel' : cg, 'tol' : 1e-10 } )

        for kwargs in opts:
            residuals = profile_solver(ml, **kwargs)

    def test_to_type(self):
        w = 1.2
        x = ones((5,1))
        y = rand(3,2)
        z = csr_matrix(rand(2,2))
        inlist = [w, x, y, z]

        out = to_type(complex, inlist)
        for i in range(len(out)):
            assert( out[i].dtype==complex ) 
            if isspmatrix(out[i]):
                diff = ravel(out[i].data - inlist[i].data)
            else:
                diff = out[i] - inlist[i]
            assert_equal( max(abs(ravel(diff))), 0.0)

    def test_type_prep(self):
        w = 1.2
        x = ones((5,1))
        y = rand(3,2)
        z = csr_matrix(rand(2,2))
        inlist = [w, x, y, z]

        out = type_prep(complex, inlist)
        for i in range(len(out)):
            assert( out[i].dtype==complex ) 
            assert( not isscalar(out[i]) )
            if isspmatrix(out[i]):
                diff = ravel(out[i].data - inlist[i].data)
            else:
                diff = out[i] - inlist[i]
            assert_equal( max(abs(ravel(diff))), 0.0)

