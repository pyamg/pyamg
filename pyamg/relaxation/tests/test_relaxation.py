from pyamg.testing import *

import numpy
import scipy
from scipy.sparse import spdiags, csr_matrix
from scipy import arange, ones, zeros, array, allclose, zeros_like, \
        tril, diag, triu, rand, asmatrix, mat
from scipy.linalg import solve

from pyamg.gallery    import poisson
from pyamg.relaxation import *

# Ignore efficiency warnings
import warnings
from scipy.sparse import SparseEfficiencyWarning
warnings.simplefilter('ignore',SparseEfficiencyWarning)

class TestCommonRelaxation(TestCase):
    def setUp(self):
        self.cases = []
        self.cases.append( (gauss_seidel,          (),               {}) )
        self.cases.append( (jacobi,                (),               {}) )
        self.cases.append( (kaczmarz_jacobi,       (),               {}) )
        self.cases.append( (kaczmarz_richardson,   (),               {}) )
        self.cases.append( (sor,                   (0.5,),           {}) )
        self.cases.append( (gauss_seidel_indexed,  ([1,0],),         {}) )
        self.cases.append( (polynomial,            ([0.6,0.1],),     {}) )


    def test_single_precision(self):

        for method,args,kwargs in self.cases:
            A = poisson((4,), format='csr').astype('float32')
            b = arange(A.shape[0], dtype='float32')
            x = 0*b
            method(A, x, b, *args, **kwargs)


    def test_double_precision(self):

        for method,args,kwargs in self.cases:
            A = poisson((4,), format='csr').astype('float64')
            b = arange(A.shape[0], dtype='float64')
            x = 0*b
            method(A, x, b, *args, **kwargs)


    def test_strided_x(self):
        """non-contiguous x should raise errors"""

        for method,args,kwargs in self.cases:
            A = poisson((4,), format='csr').astype('float64')
            b = arange(A.shape[0], dtype='float64')
            x = zeros(2*A.shape[0])[::2]
            assert_raises(ValueError, method, A, x, b, *args, **kwargs)


    def test_mixed_precision(self):
        """mixed precision arguments should raise errors"""

        for method,args,kwargs in self.cases:
            A32 = poisson((4,), format='csr').astype('float32')
            b32 = arange(A32.shape[0], dtype='float32')
            x32 = 0*b32
            
            A64 = poisson((4,), format='csr').astype('float64')
            b64 = arange(A64.shape[0], dtype='float64')
            x64 = 0*b64

            assert_raises(TypeError, method, A32, x32, b64, *args, **kwargs)
            assert_raises(TypeError, method, A32, x64, b32, *args, **kwargs)
            assert_raises(TypeError, method, A64, x32, b32, *args, **kwargs)
            assert_raises(TypeError, method, A32, x64, b64, *args, **kwargs)
            assert_raises(TypeError, method, A64, x64, b32, *args, **kwargs)
            assert_raises(TypeError, method, A64, x32, b64, *args, **kwargs)


    def test_vector_sizes(self):
        """incorrect vector sizes should raise errors"""

        for method,args,kwargs in self.cases:
            A = poisson((4,), format='csr').astype('float64')
            b4 = arange(4, dtype='float64')
            x4 = 0*b4
            b5 = arange(5, dtype='float64')
            x5 = 0*b5
            
            assert_raises(ValueError, method, A, x4, b5, *args, **kwargs)
            assert_raises(ValueError, method, A, x5, b4, *args, **kwargs)
            assert_raises(ValueError, method, A, x5, b5, *args, **kwargs)

    def test_non_square_matrix(self):

        for method,args,kwargs in self.cases:
            A = poisson((4,), format='csr').astype('float64')
            A = A[:3]
            b = arange(A.shape[0], dtype='float64')
            x = ones(A.shape[1], dtype='float64')

            assert_raises(ValueError, method, A, x, b, *args, **kwargs)

class TestRelaxation(TestCase):
    def test_polynomial(self):
        N  = 3
        A  = spdiags([2*ones(N),-ones(N),-ones(N)],[0,-1,1],N,N,format='csr')
        x0 = arange(N, dtype=A.dtype)
        x  = x0.copy()
        b  = zeros(N, dtype=A.dtype)

        r = (b - A*x0)
        polynomial(A,x,b,[-1.0/3.0])
        assert_almost_equal(x,x0-1.0/3.0*r)

        x = x0.copy()
        polynomial(A,x,b,[0.2,-1])
        assert_almost_equal(x,x0 + 0.2*A*r - r)

        x = x0.copy()
        polynomial(A,x,b,[0.2,-1])
        assert_almost_equal(x,x0 + 0.2*A*r - r)

        x = x0.copy()
        polynomial(A,x,b,[-0.14285714,  1., -2.])
        assert_almost_equal(x,x0 - 0.14285714*A*A*r + A*r - 2*r)

        # polynomial() optimizes for the case x=0
        x = 0*x0
        polynomial(A,x,b,[-1.0/3.0])
        assert_almost_equal(x,1.0/3.0*b)

        x = 0*x0
        polynomial(A,x,b,[-0.14285714,  1., -2.])
        assert_almost_equal(x,0.14285714*A*A*b + A*b - 2*b)


    def test_jacobi(self):
        N = 1
        A = spdiags([2*ones(N),-ones(N),-ones(N)],[0,-1,1],N,N,format='csr')
        x = arange(N).astype(numpy.float64)
        b = zeros(N)
        jacobi(A,x,b)
        assert_almost_equal(x,array([0]))

        N = 3
        A = spdiags([2*ones(N),-ones(N),-ones(N)],[0,-1,1],N,N,format='csr')
        x = zeros(N)
        b = arange(N).astype(numpy.float64)
        jacobi(A,x,b)
        assert_almost_equal(x,array([0.0,0.5,1.0]))

        N = 3
        A = spdiags([2*ones(N),-ones(N),-ones(N)],[0,-1,1],N,N,format='csr')
        x = arange(N).astype(numpy.float64)
        b = zeros(N)
        jacobi(A,x,b)
        assert_almost_equal(x,array([0.5,1.0,0.5]))

        N = 1
        A = spdiags([2*ones(N),-ones(N),-ones(N)],[0,-1,1],N,N,format='csr')
        x = arange(N, dtype=A.dtype)
        b = array([10], dtype=A.dtype)
        jacobi(A,x,b)
        assert_almost_equal(x,array([5]))

        N = 3
        A = spdiags([2*ones(N),-ones(N),-ones(N)],[0,-1,1],N,N,format='csr')
        x = arange(N, dtype=A.dtype)
        b = array([10,20,30], dtype=A.dtype)
        jacobi(A,x,b)
        assert_almost_equal(x,array([5.5,11.0,15.5]))

        N = 3
        A = spdiags([2*ones(N),-ones(N),-ones(N)],[0,-1,1],N,N,format='csr')
        x = arange(N, dtype=A.dtype)
        x_copy = x.copy()
        b = array([10,20,30], dtype=A.dtype)
        jacobi(A,x,b,omega=1.0/3.0)
        assert_almost_equal(x,2.0/3.0*x_copy + 1.0/3.0*array([5.5,11.0,15.5]))

    def test_gauss_seidel_bsr(self):
        cases = []

        for N in [1,2,3,4,5,6,10]:
            A = spdiags([2*ones(N),-ones(N),-ones(N)],[0,-1,1],N,N).tocsr()
            
            divisors = [ n for n in range(1,N+1) if N % n == 0 ]

            x_csr = arange(N).astype(numpy.float64)
            b = x_csr**2
            gauss_seidel(A,x_csr,b)

            for D in divisors:
                B = A.tobsr(blocksize=(D,D))
                x_bsr = arange(N).astype(numpy.float64)
                gauss_seidel(B,x_bsr,b)
                assert_almost_equal(x_bsr,x_csr)
               
    def test_gauss_seidel_gold(self):
        scipy.random.seed(0)

        cases = []
        cases.append( poisson( (4,), format='csr' ) )
        cases.append( poisson( (4,4), format='csr' ) )

        temp = asmatrix( rand(4,4) )
        cases.append( csr_matrix( temp.T * temp) )

        # reference implementation
        def gold(A,x,b,iterations,sweep):
            A = A.todense()

            L = tril(A,k=-1)
            D = diag(diag(A))
            U = triu(A,k=1)

            for i in range(iterations):
                if sweep == 'forward':
                    x = solve(L + D, (b - U*x) )
                elif sweep == 'backward':
                    x = solve(U + D, (b - L*x) )
                else:
                    x = solve(L + D, (b - U*x) )
                    x = solve(U + D, (b - L*x) )
            return x            


        for A in cases:

            b = asmatrix(rand(A.shape[0],1))
            x = asmatrix(rand(A.shape[0],1))

            x_copy = x.copy()
            gauss_seidel(A, x, b, iterations=1, sweep='forward')
            assert_almost_equal( x, gold(A,x_copy,b,iterations=1,sweep='forward') )
            
            x_copy = x.copy()
            gauss_seidel(A, x, b, iterations=1, sweep='backward')
            assert_almost_equal( x, gold(A,x_copy,b,iterations=1,sweep='backward') )
            
            x_copy = x.copy()
            gauss_seidel(A, x, b, iterations=1, sweep='symmetric')
            assert_almost_equal( x, gold(A,x_copy,b,iterations=1,sweep='symmetric') )



    def test_gauss_seidel_csr(self):
        N = 1
        A = spdiags([2*ones(N),-ones(N),-ones(N)],[0,-1,1],N,N,format='csr')
        x = arange(N).astype(numpy.float64)
        b = zeros(N)
        gauss_seidel(A,x,b)
        assert_almost_equal(x,array([0]))

        N = 3
        A = spdiags([2*ones(N),-ones(N),-ones(N)],[0,-1,1],N,N,format='csr')
        x = arange(N).astype(numpy.float64)
        b = zeros(N)
        gauss_seidel(A,x,b)
        assert_almost_equal(x,array([1.0/2.0,5.0/4.0,5.0/8.0]))

        N = 1
        A = spdiags([2*ones(N),-ones(N),-ones(N)],[0,-1,1],N,N,format='csr')
        x = arange(N).astype(numpy.float64)
        b = zeros(N)
        gauss_seidel(A,x,b,sweep='backward')
        assert_almost_equal(x,array([0]))

        N = 3
        A = spdiags([2*ones(N),-ones(N),-ones(N)],[0,-1,1],N,N,format='csr')
        x = arange(N).astype(numpy.float64)
        b = zeros(N)
        gauss_seidel(A,x,b,sweep='backward')
        assert_almost_equal(x,array([1.0/8.0,1.0/4.0,1.0/2.0]))

        N = 1
        A = spdiags([2*ones(N),-ones(N),-ones(N)],[0,-1,1],N,N,format='csr')
        x = arange(N, dtype=A.dtype)
        b = array([10], dtype=A.dtype)
        gauss_seidel(A,x,b)
        assert_almost_equal(x,array([5]))

        N = 3
        A = spdiags([2*ones(N),-ones(N),-ones(N)],[0,-1,1],N,N,format='csr')
        x = arange(N, dtype=A.dtype)
        b = array([10,20,30], dtype=A.dtype)
        gauss_seidel(A,x,b)
        assert_almost_equal(x,array([11.0/2.0,55.0/4,175.0/8.0]))


        #forward and backward passes should give same result with x=ones(N),b=zeros(N)
        N = 100
        A = spdiags([2*ones(N),-ones(N),-ones(N)],[0,-1,1],N,N,format='csr')
        x = ones(N)
        b = zeros(N)
        gauss_seidel(A,x,b,iterations=200,sweep='forward')
        resid1 = numpy.linalg.norm(A*x,2)
        x = ones(N)
        gauss_seidel(A,x,b,iterations=200,sweep='backward')
        resid2 = numpy.linalg.norm(A*x,2)
        self.assert_(resid1 < 0.01 and resid2 < 0.01)
        self.assert_(allclose(resid1,resid2))
    
    def test_gauss_seidel_indexed(self):
        N = 1
        A = spdiags([2*ones(N),-ones(N),-ones(N)],[0,-1,1],N,N,format='csr')
        x = arange(N).astype(numpy.float64)
        b = zeros(N)
        gauss_seidel_indexed(A,x,b,[0])
        assert_almost_equal(x,array([0]))

        N = 3
        A = spdiags([2*ones(N),-ones(N),-ones(N)],[0,-1,1],N,N,format='csr')
        x = arange(N).astype(numpy.float64)
        b = zeros(N)
        gauss_seidel_indexed(A,x,b,[0,1,2])
        assert_almost_equal(x,array([1.0/2.0,5.0/4.0,5.0/8.0]))
        
        N = 3
        A = spdiags([2*ones(N),-ones(N),-ones(N)],[0,-1,1],N,N,format='csr')
        x = arange(N).astype(numpy.float64)
        b = zeros(N)
        gauss_seidel_indexed(A,x,b,[2,1,0],sweep='backward')
        assert_almost_equal(x,array([1.0/2.0,5.0/4.0,5.0/8.0]))
        
        N = 3
        A = spdiags([2*ones(N),-ones(N),-ones(N)],[0,-1,1],N,N,format='csr')
        x = arange(N).astype(numpy.float64)
        b = zeros(N)
        gauss_seidel_indexed(A,x,b,[0,1,2],sweep='backward')
        assert_almost_equal(x,array([1.0/8.0,1.0/4.0,1.0/2.0]))
        
        N = 4
        A = spdiags([2*ones(N),-ones(N),-ones(N)],[0,-1,1],N,N,format='csr')
        x = ones(N)
        b = zeros(N)
        gauss_seidel_indexed(A,x,b,[0,3])
        assert_almost_equal(x,array([1.0/2.0, 1.0, 1.0, 1.0/2.0]))
        
        N = 4
        A = spdiags([2*ones(N),-ones(N),-ones(N)],[0,-1,1],N,N,format='csr')
        x = ones(N)
        b = zeros(N)
        gauss_seidel_indexed(A,x,b,[0,0])
        assert_almost_equal(x,array([1.0/2.0, 1.0, 1.0, 1.0]))

    def test_kaczmarz_jacobi(self):
        N = 1
        A = spdiags([2*ones(N),-ones(N),-ones(N)],[0,-1,1],N,N,format='csr')
        x = arange(N).astype(numpy.float64)
        b = zeros(N)
        kaczmarz_jacobi(A,x,b)
        assert_almost_equal(x,array([0]))

        N = 3
        A = spdiags([2*ones(N),-ones(N),-ones(N)],[0,-1,1],N,N,format='csr')
        x = zeros(N)
        b = arange(N).astype(numpy.float64)
        kaczmarz_jacobi(A,x,b)
        assert_almost_equal(x,array([-1./6., -1./15., 19./30.]))

        N = 3
        A = spdiags([2*ones(N),-ones(N),-ones(N)],[0,-1,1],N,N,format='csr')
        x = arange(N).astype(numpy.float64)
        b = zeros(N)
        kaczmarz_jacobi(A,x,b)
        assert_almost_equal(x,array([2./5., 7./5., 4./5.]))

        N = 1
        A = spdiags([2*ones(N),-ones(N),-ones(N)],[0,-1,1],N,N,format='csr')
        x = arange(N, dtype=A.dtype)
        b = array([10], dtype=A.dtype)
        kaczmarz_jacobi(A,x,b)
        assert_almost_equal(x,array([5]))

        N = 3
        A = spdiags([2*ones(N),-ones(N),-ones(N)],[0,-1,1],N,N,format='csr')
        x = arange(N, dtype=A.dtype)
        b = array([10,20,30], dtype=A.dtype)
        kaczmarz_jacobi(A,x,b)
        assert_almost_equal(x,array([16./15., 1./15., (9 + 7./15.) ]))

        N = 3
        A = spdiags([2*ones(N),-ones(N),-ones(N)],[0,-1,1],N,N,format='csr')
        x = arange(N, dtype=A.dtype)
        x_copy = x.copy()
        b = array([10,20,30], dtype=A.dtype)
        kaczmarz_jacobi(A,x,b,omega=1.0/3.0)
        assert_almost_equal(x,2.0/3.0*x_copy + 1.0/3.0*array([16./15., 1./15., (9 + 7./15.) ]))

    def test_kaczmarz_richardson(self):
        N = 1
        A = spdiags([2*ones(N),-ones(N),-ones(N)],[0,-1,1],N,N,format='csr')
        x = arange(N).astype(numpy.float64)
        b = zeros(N)
        kaczmarz_richardson(A,x,b)
        assert_almost_equal(x,array([0.]))

        N = 3
        A = spdiags([2*ones(N),-ones(N),-ones(N)],[0,-1,1],N,N,format='csr')
        x = zeros(N)
        b = arange(N).astype(numpy.float64)
        kaczmarz_richardson(A,x,b)
        assert_almost_equal(x,array([-1., 0., 3.]))

        N = 3
        A = spdiags([2*ones(N),-ones(N),-ones(N)],[0,-1,1],N,N,format='csr')
        x = arange(N).astype(numpy.float64)
        b = zeros(N)
        kaczmarz_richardson(A,x,b)
        assert_almost_equal(x,array([2., 3., -4.]))

        N = 1
        A = spdiags([2*ones(N),-ones(N),-ones(N)],[0,-1,1],N,N,format='csr')
        x = arange(N, dtype=A.dtype)
        b = array([10], dtype=A.dtype)
        kaczmarz_richardson(A,x,b)
        assert_almost_equal(x,array([20.]))

        N = 3
        A = spdiags([2*ones(N),-ones(N),-ones(N)],[0,-1,1],N,N,format='csr')
        x = arange(N, dtype=A.dtype)
        b = array([10,20,30], dtype=A.dtype)
        kaczmarz_richardson(A,x,b)
        assert_almost_equal(x,array([2., 3., 36.]))

        N = 3
        A = spdiags([2*ones(N),-ones(N),-ones(N)],[0,-1,1],N,N,format='csr')
        x = arange(N, dtype=A.dtype)
        x_copy = x.copy()
        b = array([10,20,30], A.dtype)
        kaczmarz_richardson(A,x,b,omega=1.0/3.0)
        assert_almost_equal(x,2.0/3.0*x_copy + 1.0/3.0*array([2., 3., 36.]))

    def test_kaczmarz_gauss_seidel_bsr(self):
        cases = []

        for N in [1,2,3,4,5,6,10]:
            A = spdiags([2*ones(N),-ones(N),-ones(N)],[0,-1,1],N,N).tocsr()
            
            divisors = [ n for n in range(1,N+1) if N % n == 0 ]

            x_csr = arange(N).astype(numpy.float64)
            b = x_csr**2
            kaczmarz_gauss_seidel(A,x_csr,b)

            for D in divisors:
                B = A.tobsr(blocksize=(D,D))
                x_bsr = arange(N).astype(numpy.float64)
                kaczmarz_gauss_seidel(B,x_bsr,b)
                assert_almost_equal(x_bsr,x_csr)
               
    def test_kaczmarz_gauss_seidel_new(self):
        scipy.random.seed(0)

        cases = []
        cases.append( poisson( (4,), format='csr' ) )
        cases.append( poisson( (4,4), format='csr' ) )

        temp = asmatrix( rand(4,4) )
        cases.append( csr_matrix( temp.T * temp) )

        # reference implementation
        def gold(A,x,b,iterations,sweep):
            A = mat(A.todense())
            AA = A*A.T

            L = tril(AA,k=0)
            U = triu(AA,k=0)

            for i in range(iterations):
                if sweep == 'forward':
                    x = x + A.T*(solve(L, (b - A*x) ))
                elif sweep == 'backward':
                    x = x + A.T*(solve(U, (b - A*x) ))
                else:
                    x = x + A.T*(solve(L, (b - A*x) ))
                    x = x + A.T*(solve(U, (b - A*x) ))
            return x            
        
        for A in cases:

            b = asmatrix(rand(A.shape[0],1))
            x = asmatrix(rand(A.shape[0],1))

            x_copy = x.copy()
            kaczmarz_gauss_seidel(A, x, b, iterations=1, sweep='forward')
            assert_almost_equal( x, gold(A,x_copy,b,iterations=1,sweep='forward') )
            
            x_copy = x.copy()
            kaczmarz_gauss_seidel(A, x, b, iterations=1, sweep='backward')
            assert_almost_equal( x, gold(A,x_copy,b,iterations=1,sweep='backward') )
            
            x_copy = x.copy()
            kaczmarz_gauss_seidel(A, x, b, iterations=1, sweep='symmetric')
            assert_almost_equal( x, gold(A,x_copy,b,iterations=1,sweep='symmetric') )



    def test_kaczmarz_gauss_seidel_csr(self):
        N = 1
        A = spdiags([2*ones(N),-ones(N),-ones(N)],[0,-1,1],N,N,format='csr')
        x = arange(N).astype(numpy.float64)
        b = zeros(N)
        kaczmarz_gauss_seidel(A,x,b)
        assert_almost_equal(x,array([0]))

        N = 3
        A = spdiags([2*ones(N),-ones(N),-ones(N)],[0,-1,1],N,N,format='csr')
        x = arange(N).astype(numpy.float64)
        b = zeros(N)
        kaczmarz_gauss_seidel(A,x,b)
        assert_almost_equal(x,array([4./15., 8./5., 4./5.]))

        N = 1
        A = spdiags([2*ones(N),-ones(N),-ones(N)],[0,-1,1],N,N,format='csr')
        x = arange(N, dtype=A.dtype)
        b = zeros(N, dtype=A.dtype)
        kaczmarz_gauss_seidel(A,x,b,sweep='backward')
        assert_almost_equal(x,array([0]))

        N = 3
        A = spdiags([2*ones(N),-ones(N),-ones(N)],[0,-1,1],N,N,format='csr')
        x = arange(N, dtype=A.dtype)
        b = zeros(N, dtype=A.dtype)
        kaczmarz_gauss_seidel(A,x,b,sweep='backward')
        assert_almost_equal(x,array([2./5., 4./5., 6./5.]))

        N = 1
        A = spdiags([2*ones(N),-ones(N),-ones(N)],[0,-1,1],N,N,format='csr')
        x = arange(N, dtype=A.dtype)
        b = array([10], dtype=A.dtype)
        kaczmarz_gauss_seidel(A,x,b)
        assert_almost_equal(x,array([5]))
        
        N = 1
        A = spdiags([2*ones(N),-ones(N),-ones(N)],[0,-1,1],N,N,format='csr')
        x = arange(N, dtype=A.dtype)
        b = array([10], dtype=A.dtype)
        kaczmarz_gauss_seidel(A,x,b,sweep='backward')
        assert_almost_equal(x,array([5]))
        
        N = 3
        A = spdiags([2*ones(N),-ones(N),-ones(N)],[0,-1,1],N,N,format='csr')
        x = arange(N, dtype=A.dtype)
        b = array([10,20,30], dtype=A.dtype)
        kaczmarz_gauss_seidel(A,x,b)
        assert_almost_equal(x,array([-2./5., -2./5., (14 + 4./5.)]))

        #forward and backward passes should give same result with x=ones(N),b=zeros(N)
        N = 100
        A = spdiags([2*ones(N),-ones(N),-ones(N)],[0,-1,1],N,N,format='csr')
        x = ones(N)
        b = zeros(N)
        kaczmarz_gauss_seidel(A,x,b,iterations=200,sweep='forward')
        resid1 = numpy.linalg.norm(A*x,2)
        x = ones(N)
        kaczmarz_gauss_seidel(A,x,b,iterations=200,sweep='backward')
        resid2 = numpy.linalg.norm(A*x,2)
        self.assert_(resid1 < 0.2 and resid2 < 0.2)
        self.assert_(allclose(resid1,resid2))


# Test complex arithmetic
class TestComplexRelaxation(TestCase):
    def test_jacobi(self):
        N = 1
        A = spdiags([2*ones(N),-ones(N),-ones(N)],[0,-1,1],N,N,format='csr')
        A.data = A.data + 1.0j*A.data
        x = arange(N).astype(A.dtype)
        b = zeros(N).astype(A.dtype)  
        jacobi(A,x,b)
        assert_almost_equal(x,array([0]))

        x = array([1.0 + 1.0j])
        b = array([-1.0 + 1.0j])
        omega = 4.0 - 1.0j
        jacobi(A,x,b,omega=omega)
        assert_almost_equal(x,array([-3.5]))

        N = 3
        A = spdiags([2*ones(N),-ones(N),-ones(N)],[0,-1,1],N,N,format='csr')
        A.data = A.data + 1.0j*A.data
        x = zeros(N).astype(A.dtype)
        b = arange(N).astype(A.dtype)
        b = b + 1.0j*b
        soln = array([0.0,0.5,1.0])
        jacobi(A,x,b)
        assert_almost_equal(x,soln)

        N = 3
        A = spdiags([2*ones(N),-ones(N),-ones(N)],[0,-1,1],N,N,format='csr')
        A.data = A.data + 1.0j*A.data
        x = arange(N).astype(A.dtype)
        x = x + 1.0j*x
        b = zeros(N).astype(A.dtype)
        soln = array([0.5 + 0.5j,1.0 + 1.0j,0.5+0.5j])
        jacobi(A,x,b)
        assert_almost_equal(x,soln)

        N = 1
        A = spdiags([2*ones(N),-ones(N),-ones(N)],[0,-1,1],N,N,format='csr')
        A.data = A.data + 1.0j*A.data
        x = arange(N).astype(A.dtype)
        b = array([10]).astype(A.dtype)
        jacobi(A,x,b)
        assert_almost_equal(x,array([2.5 - 2.5j]))

        N = 3
        A = spdiags([2*ones(N),-ones(N),-ones(N)],[0,-1,1],N,N,format='csr')
        A.data = A.data + 1.0j*A.data
        x = arange(N).astype(A.dtype)
        x = x + 1.0j*x
        b = array([10,20,30]).astype(A.dtype)
        soln = array([3.0-2.0j, 6.0-4.0j, 8.0-7.0j])
        jacobi(A,x,b)
        assert_almost_equal(x,soln)

        N = 3
        A = spdiags([2*ones(N),-ones(N),-ones(N)],[0,-1,1],N,N,format='csr')
        A.data = A.data + 1.0j*A.data
        x = arange(N).astype(A.dtype)
        x = x + 1.0j*x
        x_copy = x.copy()
        b = array([10,20,30]).astype(A.dtype)
        soln = 2.0/3.0*x_copy + 1.0/3.0*array([3.0-2.0j, 6.0-4.0j, 8.0-7.0j])
        jacobi(A,x,b,omega=1.0/3.0)
        assert_almost_equal(x,soln)
        
    def test_gauss_seidel_bsr(self):
        cases = []

        for N in [1,2,3,4,5,6,10]:
            A = spdiags([2*ones(N),-ones(N),-ones(N)],[0,-1,1],N,N).tocsr()
            A.data = A.data + 1.0j*1e-3*rand(A.data.shape[0],) 
            
            divisors = [ n for n in range(1,N+1) if N % n == 0 ]

            x_csr = (arange(N) + 1.0j*1e-3*rand(N,)).astype(A.dtype)
            x_bsr = x_csr.copy()
            b = x_csr**2
            gauss_seidel(A,x_csr,b)

            for D in divisors:
                B = A.tobsr(blocksize=(D,D))
                x_bsr_temp = x_bsr.copy()
                gauss_seidel(B,x_bsr_temp,b)
                assert_almost_equal(x_bsr_temp,x_csr)

    def test_gauss_seidel_new(self):
        scipy.random.seed(0)

        cases = []
        A = poisson( (4,), format='csr' ); A.data = A.data + 1.0j*1e-3*rand(A.data.shape[0],)
        cases.append(A)
        A = poisson( (4,4), format='csr' ); A.data = A.data + 1.0j*1e-3*rand(A.data.shape[0],)
        cases.append(A); 

        temp = asmatrix( rand(4,4) ) + 1.0j*asmatrix( rand(4,4) )
        cases.append( csr_matrix( temp.H * temp) )

        # reference implementation
        def gold(A,x,b,iterations,sweep):
            A = A.todense()

            L = tril(A,k=-1)
            D = diag(diag(A))
            U = triu(A,k=1)

            for i in range(iterations):
                if sweep == 'forward':
                    x = solve(L + D, (b - U*x) )
                elif sweep == 'backward':
                    x = solve(U + D, (b - L*x) )
                else:
                    x = solve(L + D, (b - U*x) )
                    x = solve(U + D, (b - L*x) )
            return x            


        for A in cases:

            b = asmatrix(rand(A.shape[0],1)) + 1.0j*asmatrix(rand(A.shape[0],1)).astype(A.dtype)
            x = asmatrix(rand(A.shape[0],1)) + 1.0j*asmatrix(rand(A.shape[0],1)).astype(A.dtype)

            x_copy = x.copy()
            gauss_seidel(A, x, b, iterations=1, sweep='forward')
            assert_almost_equal( x, gold(A,x_copy,b,iterations=1,sweep='forward') )
            
            x_copy = x.copy()
            gauss_seidel(A, x, b, iterations=1, sweep='backward')
            assert_almost_equal( x, gold(A,x_copy,b,iterations=1,sweep='backward') )
            
            x_copy = x.copy()
            gauss_seidel(A, x, b, iterations=1, sweep='symmetric')
            assert_almost_equal( x, gold(A,x_copy,b,iterations=1,sweep='symmetric') )

    def test_gauss_seidel_csr(self):
        N = 1
        A = spdiags([2*ones(N),-ones(N),-ones(N)],[0,-1,1],N,N,format='csr')
        A.data = A.data + 1.0j*A.data
        x = arange(N).astype(A.dtype)
        x = x + 1.0j*x
        b = zeros(N).astype(A.dtype)
        gauss_seidel(A,x,b)
        assert_almost_equal(x,array([0]))

        N = 3
        A = spdiags([2*ones(N),-ones(N),-ones(N)],[0,-1,1],N,N,format='csr')
        A.data = A.data + 1.0j*A.data
        x = arange(N).astype(A.dtype)
        x = x + 1.0j*x
        soln = array([1.0/2.0,5.0/4.0,5.0/8.0])
        soln = soln + 1.0j*soln
        b = zeros(N).astype(A.dtype)
        gauss_seidel(A,x,b)
        assert_almost_equal(x,soln)
        
        N = 1
        A = spdiags([2*ones(N),-ones(N),-ones(N)],[0,-1,1],N,N,format='csr')
        A.data = A.data + 1.0j*A.data
        x = arange(N).astype(A.dtype)
        x = x + 1.0j*x
        b = zeros(N).astype(A.dtype)
        gauss_seidel(A,x,b,sweep='backward')
        assert_almost_equal(x,array([0]))

        N = 3
        A = spdiags([2*ones(N),-ones(N),-ones(N)],[0,-1,1],N,N,format='csr')
        A.data = A.data + 1.0j*A.data
        x = arange(N).astype(A.dtype)
        x = x + 1.0j*x
        soln = array([1.0/8.0,1.0/4.0,1.0/2.0])
        soln = soln + 1.0j*soln
        b = zeros(N).astype(A.dtype)
        gauss_seidel(A,x,b,sweep='backward')
        assert_almost_equal(x,soln)

        N = 1
        A = spdiags([2*ones(N),-ones(N),-ones(N)],[0,-1,1],N,N,format='csr')
        A.data = A.data + 1.0j*A.data
        x = arange(N).astype(A.dtype)
        b = array([10]).astype(A.dtype)
        gauss_seidel(A,x,b)
        assert_almost_equal(x,array([2.5 - 2.5j]))

        N = 3
        A = spdiags([2*ones(N),-ones(N),-ones(N)],[0,-1,1],N,N,format='csr')
        A.data = A.data + 1.0j*A.data
        soln = array([3.0 - 2.0j, 7.5 - 5.0j, 11.25 - 10.0j])
        x = arange(N).astype(A.dtype)
        x = x + 1.0j*x
        b = array([10,20,30]).astype(A.dtype)
        gauss_seidel(A,x,b)
        assert_almost_equal(x,soln)
        
        #forward and backward passes should give same result with x=ones(N),b=zeros(N)
        N = 100
        A = spdiags([2*ones(N),-ones(N),-ones(N)],[0,-1,1],N,N,format='csr')
        A.data = A.data + 1.0j*A.data
        x = ones(N).astype(A.dtype)
        x = x + 1.0j*x
        b = zeros(N).astype(A.dtype)
        gauss_seidel(A,x,b,iterations=200,sweep='forward')
        resid1 = numpy.linalg.norm(A*x,2)
        x = ones(N).astype(A.dtype)
        x = x + 1.0j*x
        gauss_seidel(A,x,b,iterations=200,sweep='backward')
        resid2 = numpy.linalg.norm(A*x,2)
        self.assert_(resid1 < 0.03 and resid2 < 0.03)
        self.assert_(allclose(resid1,resid2))

    def test_kaczmarz_jacobi(self):
        N = 1
        A = spdiags([2*ones(N),-ones(N),-ones(N)],[0,-1,1],N,N,format='csr')
        A.data = A.data + 1.0j*A.data
        x = arange(N).astype(A.dtype)
        b = zeros(N).astype(A.dtype)
        kaczmarz_jacobi(A,x,b)
        assert_almost_equal(x,array([0]))

        N = 3
        A = spdiags([2*ones(N),-ones(N),-ones(N)],[0,-1,1],N,N,format='csr')
        A.data = A.data + 1.0j*A.data
        soln = array([-1./6., -1./15., 19./30.])
        x = zeros(N).astype(A.dtype)
        b = arange(N).astype(A.dtype)
        b = b + 1.0j*b
        kaczmarz_jacobi(A,x,b)
        assert_almost_equal(x,soln)

        N = 3
        A = spdiags([2*ones(N),-ones(N),-ones(N)],[0,-1,1],N,N,format='csr')
        A.data = A.data + 1.0j*A.data
        soln = array([2./5. + 2.0j/5., 7./5. + 7.0j/5., 4./5. + 4.0j/5.])
        x = arange(N).astype(A.dtype)
        x = x + 1.0j*x
        b = zeros(N).astype(A.dtype)
        kaczmarz_jacobi(A,x,b)
        assert_almost_equal(x,soln)

        N = 1
        A = spdiags([2*ones(N),-ones(N),-ones(N)],[0,-1,1],N,N,format='csr')
        A.data = A.data + 1.0j*A.data
        x = arange(N).astype(A.dtype)
        b = array([10]).astype(A.dtype)
        kaczmarz_jacobi(A,x,b)
        assert_almost_equal(x,array([2.5 - 2.5j]))

        N = 3
        A = spdiags([2*ones(N),-ones(N),-ones(N)],[0,-1,1],N,N,format='csr')
        A.data = A.data + 1.0j*A.data
        soln = array([11./15. + 1.0j/15., 11./15. + 31.0j/15, 77./15. - 53.0j/15.])
        x = arange(N).astype(A.dtype)
        x = x + 1.0j*x
        b = array([10,20,30]).astype(A.dtype)
        kaczmarz_jacobi(A,x,b)
        assert_almost_equal(x,soln)

        N = 3
        A = spdiags([2*ones(N),-ones(N),-ones(N)],[0,-1,1],N,N,format='csr')
        A.data = A.data + 1.0j*A.data
        x = arange(N).astype(A.dtype)
        x = x + 1.0j*x
        b = array([10,20,30]).astype(A.dtype)
        x_copy = x.copy()
        soln = 2.0/3.0*x_copy + 1.0/3.0*array([11./15. + 1.0j/15., 11./15. + 31.0j/15, 77./15. - 53.0j/15.])
        kaczmarz_jacobi(A,x,b,omega=1.0/3.0)
        assert_almost_equal(x,soln)

    def test_kaczmarz_gauss_seidel_bsr(self):
        cases = []

        for N in [1,2,3,4,5,6,10]:
            A = spdiags([2*ones(N),-ones(N),-ones(N)],[0,-1,1],N,N).tocsr()
            A.data = A.data + 1.0j*1e-3*rand(A.data.shape[0],) 
            
            divisors = [ n for n in range(1,N+1) if N % n == 0 ]

            x_csr = (arange(N) + 1.0j*1e-3*rand(N,)).astype(A.dtype)
            x_bsr = x_csr.copy()
            b = x_csr**2
            kaczmarz_gauss_seidel(A,x_csr,b)

            for D in divisors:
                B = A.tobsr(blocksize=(D,D))
                x_bsr_temp = x_bsr.copy()
                kaczmarz_gauss_seidel(B,x_bsr_temp,b)
                assert_almost_equal(x_bsr_temp,x_csr)
               
    def test_kaczmarz_gauss_seidel_new(self):
        scipy.random.seed(0)

        cases = []
        A = poisson( (4,), format='csr' ); A.data = A.data + 1.0j*1e-3*rand(A.data.shape[0],)
        cases.append(A)
        A = poisson( (4,4), format='csr' ); A.data = A.data + 1.0j*1e-3*rand(A.data.shape[0],)
        cases.append(A.tobsr(blocksize=(2,2))) 

        temp = asmatrix( rand(4,4) ) + 1.0j*asmatrix( rand(4,4) )
        cases.append( csr_matrix( temp.T * temp) )

        # reference implementation
        def gold(A,x,b,iterations,sweep):
            A = mat(A.todense())
            AA = A*A.H

            L = tril(AA,k=0)
            U = triu(AA,k=0)

            for i in range(iterations):
                if sweep == 'forward':
                    x = x + A.H*(solve(L, (b - A*x) ))
                elif sweep == 'backward':
                    x = x + A.H*(solve(U, (b - A*x) ))
                else:
                    x = x + A.H*(solve(L, (b - A*x) ))
                    x = x + A.H*(solve(U, (b - A*x) ))
            return x            


        for A in cases:

            b = asmatrix(rand(A.shape[0],1)) + 1.0j*asmatrix(rand(A.shape[0],1)).astype(A.dtype)
            x = asmatrix(rand(A.shape[0],1)) + 1.0j*asmatrix(rand(A.shape[0],1)).astype(A.dtype)

            x_copy = x.copy()
            kaczmarz_gauss_seidel(A, x, b, iterations=1, sweep='forward')
            assert_almost_equal( x, gold(A,x_copy,b,iterations=1,sweep='forward') )
            
            x_copy = x.copy()
            kaczmarz_gauss_seidel(A, x, b, iterations=1, sweep='backward')
            assert_almost_equal( x, gold(A,x_copy,b,iterations=1,sweep='backward') )
            
            x_copy = x.copy()
            kaczmarz_gauss_seidel(A, x, b, iterations=1, sweep='symmetric')
            assert_almost_equal( x, gold(A,x_copy,b,iterations=1,sweep='symmetric') )



    def test_kaczmarz_gauss_seidel_csr(self):
        N = 1
        A = spdiags([2*ones(N),-ones(N),-ones(N)],[0,-1,1],N,N,format='csr')
        x = arange(N).astype(A.dtype)
        x = x + 1.0j*x
        A.data = A.data + 1.0j*A.data
        b = zeros(N).astype(A.dtype)
        kaczmarz_gauss_seidel(A,x,b)
        assert_almost_equal(x,array([0]))

        N = 3
        A = spdiags([2*ones(N),-ones(N),-ones(N)],[0,-1,1],N,N,format='csr')
        A.data = A.data + 1.0j*A.data
        soln = array([4./15., 8./5., 4./5.])
        soln = soln + 1.0j*soln
        x = arange(N).astype(A.dtype)
        x = x + 1.0j*x
        b = zeros(N).astype(A.dtype)
        kaczmarz_gauss_seidel(A,x,b)
        assert_almost_equal(x,soln)
        
        N = 1
        A = spdiags([2*ones(N),-ones(N),-ones(N)],[0,-1,1],N,N,format='csr')
        A.data = A.data + 1.0j*A.data
        x = arange(N).astype(A.dtype)
        x = x + 1.0j*x
        b = zeros(N).astype(A.dtype)
        kaczmarz_gauss_seidel(A,x,b,sweep='backward')
        assert_almost_equal(x,array([0]))

        N = 3
        A = spdiags([2*ones(N),-ones(N),-ones(N)],[0,-1,1],N,N,format='csr')
        A.data = A.data + 1.0j*A.data
        soln = array([2./5., 4./5., 6./5.])
        soln = soln + 1.0j*soln
        x = arange(N).astype(A.dtype)
        x = x + 1.0j*x
        b = zeros(N).astype(A.dtype)
        kaczmarz_gauss_seidel(A,x,b,sweep='backward')
        assert_almost_equal(x,soln)

        N = 1
        A = spdiags([2*ones(N),-ones(N),-ones(N)],[0,-1,1],N,N,format='csr')
        A.data = A.data + 1.0j*A.data
        x = arange(N).astype(A.dtype)
        b = array([10]).astype(A.dtype)
        kaczmarz_gauss_seidel(A,x,b)
        assert_almost_equal(x,array([2.5 - 2.5j]))

        N = 3
        A = spdiags([2*ones(N),-ones(N),-ones(N)],[0,-1,1],N,N,format='csr')
        A.data = A.data + 1.0j*A.data
        soln = array([-1./15.+0.6j,  0.6+2.6j,  7.8-6.2j])
        x = arange(N).astype(A.dtype)
        x = x + 1.0j*x
        b = array([10,20,30]).astype(A.dtype)
        kaczmarz_gauss_seidel(A,x,b)
        assert_almost_equal(x,soln)
        
        #forward and backward passes should give same result with x=ones(N),b=zeros(N)
        N = 100
        A = spdiags([2*ones(N),-ones(N),-ones(N)],[0,-1,1],N,N,format='csr')
        A.data = A.data + 1.0j*A.data
        x = ones(N).astype(A.dtype)
        x = x + 1.0j*x
        b = zeros(N).astype(A.dtype)
        kaczmarz_gauss_seidel(A,x,b,iterations=200,sweep='forward')
        resid1 = numpy.linalg.norm(A*x,2)
        
        x = ones(N).astype(A.dtype)
        x = x + 1.0j*x
        kaczmarz_gauss_seidel(A,x,b,iterations=200,sweep='backward')
        resid2 = numpy.linalg.norm(A*x,2)
        self.assert_(resid1 < 0.3 and resid2 < 0.3)
        self.assert_(allclose(resid1,resid2))


#class TestDispatch(TestCase):
#    def test_string(self):
#        from pyamg.relaxation import dispatch
#        
#        A = poisson( (4,), format='csr')
#        
#        cases = []
#        cases.append( 'gauss_seidel' )
#        cases.append( ('gauss_seidel',{'iterations':3}) )
#        
#        for case in cases:
#            fn = dispatch(case)
#            fn(A, ones(4), zeros(4))


