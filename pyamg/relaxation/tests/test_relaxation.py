import numpy as np
import scipy
from scipy.sparse import spdiags, csr_matrix, bsr_matrix, eye
from scipy import arange, ones, zeros, array, allclose, \
    diag, triu, tril, rand, asmatrix, mat
from scipy.linalg import solve

from pyamg.gallery import poisson, sprand, elasticity
from pyamg.relaxation.relaxation import gauss_seidel, jacobi,\
    block_jacobi, block_gauss_seidel, jacobi_ne, schwarz, sor,\
    gauss_seidel_indexed, polynomial, gauss_seidel_ne,\
    gauss_seidel_nr
from pyamg.util.utils import get_block_diag

from numpy.testing import TestCase, assert_almost_equal

# Ignore efficiency warnings
import warnings
from scipy.sparse import SparseEfficiencyWarning
warnings.simplefilter('ignore', SparseEfficiencyWarning)


def check_raises(error, f, *args, **kwargs):
    try:
        f(*args, **kwargs)
    except error:
        pass
    else:
        raise Exception("%s should throw an error" % f.__name__)


class TestCommonRelaxation(TestCase):
    def setUp(self):
        self.cases = []
        self.cases.append((gauss_seidel, (), {}))
        self.cases.append((jacobi, (), {}))
        self.cases.append((block_jacobi, (), {}))
        self.cases.append((block_gauss_seidel, (), {}))
        self.cases.append((jacobi_ne, (), {}))
        self.cases.append((schwarz, (), {}))
        self.cases.append((sor, (0.5,), {}))
        self.cases.append((gauss_seidel_indexed, ([1, 0],), {}))
        self.cases.append((polynomial, ([0.6, 0.1],), {}))

    def test_single_precision(self):

        for method, args, kwargs in self.cases:
            A = poisson((4,), format='csr').astype('float32')
            b = arange(A.shape[0], dtype='float32')
            x = 0*b
            method(A, x, b, *args, **kwargs)

    def test_double_precision(self):

        for method, args, kwargs in self.cases:
            A = poisson((4,), format='csr').astype('float64')
            b = arange(A.shape[0], dtype='float64')
            x = 0*b
            method(A, x, b, *args, **kwargs)

    def test_strided_x(self):
        """non-contiguous x should raise errors"""

        for method, args, kwargs in self.cases:
            A = poisson((4,), format='csr').astype('float64')
            b = arange(A.shape[0], dtype='float64')
            x = zeros(2*A.shape[0])[::2]
            check_raises(ValueError, method, A, x, b, *args, **kwargs)

    def test_mixed_precision(self):
        """mixed precision arguments should raise errors"""

        for method, args, kwargs in self.cases:
            A32 = poisson((4,), format='csr').astype('float32')
            b32 = arange(A32.shape[0], dtype='float32')
            x32 = 0*b32

            A64 = poisson((4,), format='csr').astype('float64')
            b64 = arange(A64.shape[0], dtype='float64')
            x64 = 0*b64

            check_raises(TypeError, method, A32, x32, b64, *args, **kwargs)
            check_raises(TypeError, method, A32, x64, b32, *args, **kwargs)
            check_raises(TypeError, method, A64, x32, b32, *args, **kwargs)
            check_raises(TypeError, method, A32, x64, b64, *args, **kwargs)
            check_raises(TypeError, method, A64, x64, b32, *args, **kwargs)
            check_raises(TypeError, method, A64, x32, b64, *args, **kwargs)

    def test_vector_sizes(self):
        """incorrect vector sizes should raise errors"""

        for method, args, kwargs in self.cases:
            A = poisson((4,), format='csr').astype('float64')
            b4 = arange(4, dtype='float64')
            x4 = 0*b4
            b5 = arange(5, dtype='float64')
            x5 = 0*b5

            check_raises(ValueError, method, A, x4, b5, *args, **kwargs)
            check_raises(ValueError, method, A, x5, b4, *args, **kwargs)
            check_raises(ValueError, method, A, x5, b5, *args, **kwargs)

    def test_non_square_matrix(self):

        for method, args, kwargs in self.cases:
            A = poisson((4,), format='csr').astype('float64')
            A = A[:3]
            b = arange(A.shape[0], dtype='float64')
            x = ones(A.shape[1], dtype='float64')

            check_raises(ValueError, method, A, x, b, *args, **kwargs)


class TestRelaxation(TestCase):
    def test_polynomial(self):
        N = 3
        A = spdiags([2*ones(N), -ones(N), -ones(N)], [0, -1, 1], N, N,
                    format='csr')
        x0 = arange(N, dtype=A.dtype)
        x = x0.copy()
        b = zeros(N, dtype=A.dtype)

        r = (b - A*x0)
        polynomial(A, x, b, [-1.0/3.0])
        assert_almost_equal(x, x0-1.0/3.0*r)

        x = x0.copy()
        polynomial(A, x, b, [0.2, -1])
        assert_almost_equal(x, x0 + 0.2*A*r - r)

        x = x0.copy()
        polynomial(A, x, b, [0.2, -1])
        assert_almost_equal(x, x0 + 0.2*A*r - r)

        x = x0.copy()
        polynomial(A, x, b, [-0.14285714, 1., -2.])
        assert_almost_equal(x, x0 - 0.14285714*A*A*r + A*r - 2*r)

        # polynomial() optimizes for the case x=0
        x = 0*x0
        polynomial(A, x, b, [-1.0/3.0])
        assert_almost_equal(x, 1.0/3.0*b)

        x = 0*x0
        polynomial(A, x, b, [-0.14285714, 1., -2.])
        assert_almost_equal(x, 0.14285714*A*A*b + A*b - 2*b)

    def test_jacobi(self):
        N = 1
        A = spdiags([2*ones(N), -ones(N), -ones(N)], [0, -1, 1], N, N,
                    format='csr')
        x = arange(N).astype(np.float64)
        b = zeros(N)
        jacobi(A, x, b)
        assert_almost_equal(x, array([0]))

        N = 3
        A = spdiags([2*ones(N), -ones(N), -ones(N)], [0, -1, 1], N, N,
                    format='csr')
        x = zeros(N)
        b = arange(N).astype(np.float64)
        jacobi(A, x, b)
        assert_almost_equal(x, array([0.0, 0.5, 1.0]))

        N = 3
        A = spdiags([2*ones(N), -ones(N), -ones(N)], [0, -1, 1], N, N,
                    format='csr')
        x = arange(N).astype(np.float64)
        b = zeros(N)
        jacobi(A, x, b)
        assert_almost_equal(x, array([0.5, 1.0, 0.5]))

        N = 1
        A = spdiags([2*ones(N), -ones(N), -ones(N)], [0, -1, 1], N, N,
                    format='csr')
        x = arange(N, dtype=A.dtype)
        b = array([10], dtype=A.dtype)
        jacobi(A, x, b)
        assert_almost_equal(x, array([5]))

        N = 3
        A = spdiags([2*ones(N), -ones(N), -ones(N)], [0, -1, 1], N, N,
                    format='csr')
        x = arange(N, dtype=A.dtype)
        b = array([10, 20, 30], dtype=A.dtype)
        jacobi(A, x, b)
        assert_almost_equal(x, array([5.5, 11.0, 15.5]))

        N = 3
        A = spdiags([2*ones(N), -ones(N), -ones(N)], [0, -1, 1], N, N,
                    format='csr')
        x = arange(N, dtype=A.dtype)
        x_copy = x.copy()
        b = array([10, 20, 30], dtype=A.dtype)
        jacobi(A, x, b, omega=1.0/3.0)
        assert_almost_equal(x,
                            2.0/3.0*x_copy + 1.0/3.0*array([5.5, 11.0, 15.5]))

    def test_jacobi_bsr(self):
        cases = []
        # JBS: remove some N
        for N in [1, 2, 3, 4, 5, 6, 10]:
            cases.append(spdiags([2*ones(N), -ones(N), -ones(N)],
                                 [0, -1, 1], N, N).tocsr())
            cases.append(elasticity.linear_elasticity((N, N))[0].tocsr())
            C = csr_matrix(rand(N, N))
            cases.append(C*C.H)
            C = sprand(N*2, N*2, 0.3) + eye(N*2, N*2)
            cases.append(C*C.H)

        for A in cases:
            divisors =\
                [n for n in range(1, A.shape[0]+1) if A.shape[0] % n == 0]

            x_csr = arange(A.shape[0]).astype(np.float64)
            b = x_csr**2
            jacobi(A, x_csr, b)

            for D in divisors:
                B = A.tobsr(blocksize=(D, D))
                x_bsr = arange(B.shape[0]).astype(np.float64)
                jacobi(B, x_bsr, b)
                assert_almost_equal(x_bsr, x_csr)

    def test_gauss_seidel_bsr(self):
        sweeps = ['forward', 'backward', 'symmetric']
        cases = []
        for N in [1, 2, 3, 4, 5, 6, 10]:
            cases.append(spdiags([2*ones(N), -ones(N), -ones(N)],
                                 [0, -1, 1], N, N).tocsr())
            cases.append(elasticity.linear_elasticity((N, N))[0].tocsr())
            C = csr_matrix(rand(N, N))
            cases.append(C*C.H)
            C = sprand(N*2, N*2, 0.3) + eye(N*2, N*2)
            cases.append(C*C.H)

        for A in cases:
            for sweep in sweeps:
                divisors =\
                    [n for n in range(1, A.shape[0]+1) if A.shape[0] % n == 0]

                x_csr = arange(A.shape[0]).astype(np.float64)
                b = x_csr**2
                gauss_seidel(A, x_csr, b, sweep=sweep)

                for D in divisors:
                    B = A.tobsr(blocksize=(D, D))
                    x_bsr = arange(B.shape[0]).astype(np.float64)
                    gauss_seidel(B, x_bsr, b, sweep=sweep)
                    assert_almost_equal(x_bsr, x_csr)

    def test_gauss_seidel_gold(self):
        scipy.random.seed(0)

        cases = []
        cases.append(poisson((4,), format='csr'))
        cases.append(poisson((4, 4), format='csr'))

        temp = asmatrix(rand(4, 4))
        cases.append(csr_matrix(temp.T * temp))

        # reference implementation
        def gold(A, x, b, iterations, sweep):
            A = A.todense()

            L = asmatrix(tril(A, k=-1))
            D = asmatrix(diag(diag(A)))
            U = asmatrix(triu(A, k=1))

            for i in range(iterations):
                if sweep == 'forward':
                    x = solve(L + D, (b - U*x))
                elif sweep == 'backward':
                    x = solve(U + D, (b - L*x))
                else:
                    x = solve(L + D, (b - U*x))
                    x = solve(U + D, (b - L*x))
            return x

        for A in cases:

            b = asmatrix(rand(A.shape[0], 1))
            x = asmatrix(rand(A.shape[0], 1))

            x_copy = x.copy()
            gauss_seidel(A, x, b, iterations=1, sweep='forward')
            assert_almost_equal(x, gold(A, x_copy, b, iterations=1,
                                sweep='forward'))

            x_copy = x.copy()
            gauss_seidel(A, x, b, iterations=1, sweep='backward')
            assert_almost_equal(x, gold(A, x_copy, b, iterations=1,
                                sweep='backward'))

            x_copy = x.copy()
            gauss_seidel(A, x, b, iterations=1, sweep='symmetric')
            assert_almost_equal(x, gold(A, x_copy, b, iterations=1,
                                sweep='symmetric'))

    def test_gauss_seidel_csr(self):
        N = 1
        A = spdiags([2*ones(N), -ones(N), -ones(N)], [0, -1, 1], N, N,
                    format='csr')
        x = arange(N).astype(np.float64)
        b = zeros(N)
        gauss_seidel(A, x, b)
        assert_almost_equal(x, array([0]))

        N = 3
        A = spdiags([2*ones(N), -ones(N), -ones(N)], [0, -1, 1], N, N,
                    format='csr')
        x = arange(N).astype(np.float64)
        b = zeros(N)
        gauss_seidel(A, x, b)
        assert_almost_equal(x, array([1.0/2.0, 5.0/4.0, 5.0/8.0]))

        N = 1
        A = spdiags([2*ones(N), -ones(N), -ones(N)], [0, -1, 1], N, N,
                    format='csr')
        x = arange(N).astype(np.float64)
        b = zeros(N)
        gauss_seidel(A, x, b, sweep='backward')
        assert_almost_equal(x, array([0]))

        N = 3
        A = spdiags([2*ones(N), -ones(N), -ones(N)], [0, -1, 1], N, N,
                    format='csr')
        x = arange(N).astype(np.float64)
        b = zeros(N)
        gauss_seidel(A, x, b, sweep='backward')
        assert_almost_equal(x, array([1.0/8.0, 1.0/4.0, 1.0/2.0]))

        N = 1
        A = spdiags([2*ones(N), -ones(N), -ones(N)], [0, -1, 1], N, N,
                    format='csr')
        x = arange(N, dtype=A.dtype)
        b = array([10], dtype=A.dtype)
        gauss_seidel(A, x, b)
        assert_almost_equal(x, array([5]))

        N = 3
        A = spdiags([2*ones(N), -ones(N), -ones(N)], [0, -1, 1], N, N,
                    format='csr')
        x = arange(N, dtype=A.dtype)
        b = array([10, 20, 30], dtype=A.dtype)
        gauss_seidel(A, x, b)
        assert_almost_equal(x, array([11.0/2.0, 55.0/4, 175.0/8.0]))

        # forward and backward passes should give same result with
        # x=ones(N),b=zeros(N)
        N = 100
        A = spdiags([2*ones(N), -ones(N), -ones(N)], [0, -1, 1], N, N,
                    format='csr')
        x = ones(N)
        b = zeros(N)
        gauss_seidel(A, x, b, iterations=200, sweep='forward')
        resid1 = np.linalg.norm(A*x, 2)
        x = ones(N)
        gauss_seidel(A, x, b, iterations=200, sweep='backward')
        resid2 = np.linalg.norm(A*x, 2)
        self.assertTrue(resid1 < 0.01 and resid2 < 0.01)
        self.assertTrue(allclose(resid1, resid2))

    def test_gauss_seidel_indexed(self):
        N = 1
        A = spdiags([2*ones(N), -ones(N), -ones(N)], [0, -1, 1], N, N,
                    format='csr')
        x = arange(N).astype(np.float64)
        b = zeros(N)
        gauss_seidel_indexed(A, x, b, [0])
        assert_almost_equal(x, array([0]))

        N = 3
        A = spdiags([2*ones(N), -ones(N), -ones(N)], [0, -1, 1], N, N,
                    format='csr')
        x = arange(N).astype(np.float64)
        b = zeros(N)
        gauss_seidel_indexed(A, x, b, [0, 1, 2])
        assert_almost_equal(x, array([1.0/2.0, 5.0/4.0, 5.0/8.0]))

        N = 3
        A = spdiags([2*ones(N), -ones(N), -ones(N)], [0, -1, 1], N, N,
                    format='csr')
        x = arange(N).astype(np.float64)
        b = zeros(N)
        gauss_seidel_indexed(A, x, b, [2, 1, 0], sweep='backward')
        assert_almost_equal(x, array([1.0/2.0, 5.0/4.0, 5.0/8.0]))

        N = 3
        A = spdiags([2*ones(N), -ones(N), -ones(N)], [0, -1, 1], N, N,
                    format='csr')
        x = arange(N).astype(np.float64)
        b = zeros(N)
        gauss_seidel_indexed(A, x, b, [0, 1, 2], sweep='backward')
        assert_almost_equal(x, array([1.0/8.0, 1.0/4.0, 1.0/2.0]))

        N = 4
        A = spdiags([2*ones(N), -ones(N), -ones(N)], [0, -1, 1], N, N,
                    format='csr')
        x = ones(N)
        b = zeros(N)
        gauss_seidel_indexed(A, x, b, [0, 3])
        assert_almost_equal(x, array([1.0/2.0, 1.0, 1.0, 1.0/2.0]))

        N = 4
        A = spdiags([2*ones(N), -ones(N), -ones(N)], [0, -1, 1], N, N,
                    format='csr')
        x = ones(N)
        b = zeros(N)
        gauss_seidel_indexed(A, x, b, [0, 0])
        assert_almost_equal(x, array([1.0/2.0, 1.0, 1.0, 1.0]))

    def test_jacobi_ne(self):
        N = 1
        A = spdiags([2*ones(N), -ones(N), -ones(N)], [0, -1, 1], N, N,
                    format='csr')
        x = arange(N).astype(np.float64)
        b = zeros(N)
        jacobi_ne(A, x, b)
        assert_almost_equal(x, array([0]))

        N = 3
        A = spdiags([2*ones(N), -ones(N), -ones(N)], [0, -1, 1], N, N,
                    format='csr')
        x = zeros(N)
        b = arange(N).astype(np.float64)
        jacobi_ne(A, x, b)
        assert_almost_equal(x, array([-1./6., -1./15., 19./30.]))

        N = 3
        A = spdiags([2*ones(N), -ones(N), -ones(N)], [0, -1, 1], N, N,
                    format='csr')
        x = arange(N).astype(np.float64)
        b = zeros(N)
        jacobi_ne(A, x, b)
        assert_almost_equal(x, array([2./5., 7./5., 4./5.]))

        N = 1
        A = spdiags([2*ones(N), -ones(N), -ones(N)], [0, -1, 1], N, N,
                    format='csr')
        x = arange(N, dtype=A.dtype)
        b = array([10], dtype=A.dtype)
        jacobi_ne(A, x, b)
        assert_almost_equal(x, array([5]))

        N = 3
        A = spdiags([2*ones(N), -ones(N), -ones(N)], [0, -1, 1], N, N,
                    format='csr')
        x = arange(N, dtype=A.dtype)
        b = array([10, 20, 30], dtype=A.dtype)
        jacobi_ne(A, x, b)
        assert_almost_equal(x, array([16./15., 1./15., (9 + 7./15.)]))

        N = 3
        A = spdiags([2*ones(N), -ones(N), -ones(N)], [0, -1, 1], N, N,
                    format='csr')
        x = arange(N, dtype=A.dtype)
        x_copy = x.copy()
        b = array([10, 20, 30], dtype=A.dtype)
        jacobi_ne(A, x, b, omega=1.0/3.0)
        xtrue = 2.0/3.0*x_copy + 1.0/3.0*array([16./15., 1./15., (9 + 7./15.)])
        assert_almost_equal(x, xtrue)

    def test_gauss_seidel_ne_bsr(self):
        # JBS: remove some N
        for N in [1, 2, 3, 4, 5, 6, 10]:
            A = spdiags([2*ones(N), -ones(N), -ones(N)], [0, -1, 1],
                        N, N).tocsr()

            divisors = [n for n in range(1, N+1) if N % n == 0]

            x_csr = arange(N).astype(np.float64)
            b = x_csr**2
            gauss_seidel_ne(A, x_csr, b)

            for D in divisors:
                B = A.tobsr(blocksize=(D, D))
                x_bsr = arange(N).astype(np.float64)
                gauss_seidel_ne(B, x_bsr, b)
                assert_almost_equal(x_bsr, x_csr)

    def test_gauss_seidel_ne_new(self):
        scipy.random.seed(0)

        cases = []
        cases.append(poisson((4,), format='csr'))
        cases.append(poisson((4, 4), format='csr'))

        temp = asmatrix(rand(4, 4))
        cases.append(csr_matrix(temp.T * temp))

        # reference implementation
        def gold(A, x, b, iterations, sweep):
            A = mat(A.todense())
            AA = A*A.T

            L = asmatrix(tril(AA, k=0))
            U = asmatrix(triu(AA, k=0))

            for i in range(iterations):
                if sweep == 'forward':
                    x = x + A.T*(solve(L, (b - A*x)))
                elif sweep == 'backward':
                    x = x + A.T*(solve(U, (b - A*x)))
                else:
                    x = x + A.T*(solve(L, (b - A*x)))
                    x = x + A.T*(solve(U, (b - A*x)))
            return x

        for A in cases:

            b = asmatrix(rand(A.shape[0], 1))
            x = asmatrix(rand(A.shape[0], 1))

            x_copy = x.copy()
            gauss_seidel_ne(A, x, b, iterations=1, sweep='forward')
            assert_almost_equal(x, gold(A, x_copy, b, iterations=1,
                                sweep='forward'))

            x_copy = x.copy()
            gauss_seidel_ne(A, x, b, iterations=1, sweep='backward')
            assert_almost_equal(x, gold(A, x_copy, b, iterations=1,
                                sweep='backward'))

            x_copy = x.copy()
            gauss_seidel_ne(A, x, b, iterations=1, sweep='symmetric')
            assert_almost_equal(x, gold(A, x_copy, b, iterations=1,
                                sweep='symmetric'))

    def test_gauss_seidel_ne_csr(self):
        N = 1
        A = spdiags([2*ones(N), -ones(N), -ones(N)], [0, -1, 1], N, N,
                    format='csr')
        x = arange(N).astype(np.float64)
        b = zeros(N)
        gauss_seidel_ne(A, x, b)
        assert_almost_equal(x, array([0]))

        N = 3
        A = spdiags([2*ones(N), -ones(N), -ones(N)], [0, -1, 1], N, N,
                    format='csr')
        x = arange(N).astype(np.float64)
        b = zeros(N)
        gauss_seidel_ne(A, x, b)
        assert_almost_equal(x, array([4./15., 8./5., 4./5.]))

        N = 1
        A = spdiags([2*ones(N), -ones(N), -ones(N)], [0, -1, 1], N, N,
                    format='csr')
        x = arange(N, dtype=A.dtype)
        b = zeros(N, dtype=A.dtype)
        gauss_seidel_ne(A, x, b, sweep='backward')
        assert_almost_equal(x, array([0]))

        N = 3
        A = spdiags([2*ones(N), -ones(N), -ones(N)], [0, -1, 1], N, N,
                    format='csr')
        x = arange(N, dtype=A.dtype)
        b = zeros(N, dtype=A.dtype)
        gauss_seidel_ne(A, x, b, sweep='backward')
        assert_almost_equal(x, array([2./5., 4./5., 6./5.]))

        N = 1
        A = spdiags([2*ones(N), -ones(N), -ones(N)], [0, -1, 1], N, N,
                    format='csr')
        x = arange(N, dtype=A.dtype)
        b = array([10], dtype=A.dtype)
        gauss_seidel_ne(A, x, b)
        assert_almost_equal(x, array([5]))

        N = 1
        A = spdiags([2*ones(N), -ones(N), -ones(N)], [0, -1, 1], N, N,
                    format='csr')
        x = arange(N, dtype=A.dtype)
        b = array([10], dtype=A.dtype)
        gauss_seidel_ne(A, x, b, sweep='backward')
        assert_almost_equal(x, array([5]))

        N = 3
        A = spdiags([2*ones(N), -ones(N), -ones(N)], [0, -1, 1], N, N,
                    format='csr')
        x = arange(N, dtype=A.dtype)
        b = array([10, 20, 30], dtype=A.dtype)
        gauss_seidel_ne(A, x, b)
        assert_almost_equal(x, array([-2./5., -2./5., (14 + 4./5.)]))

        # forward and backward passes should give same result with
        # x=ones(N),b=zeros(N)
        N = 100
        A = spdiags([2*ones(N), -ones(N), -ones(N)], [0, -1, 1], N, N,
                    format='csr')
        x = ones(N)
        b = zeros(N)
        gauss_seidel_ne(A, x, b, iterations=200, sweep='forward')
        resid1 = np.linalg.norm(A*x, 2)
        x = ones(N)
        gauss_seidel_ne(A, x, b, iterations=200, sweep='backward')
        resid2 = np.linalg.norm(A*x, 2)
        self.assertTrue(resid1 < 0.2 and resid2 < 0.2)
        self.assertTrue(allclose(resid1, resid2))

    def test_gauss_seidel_nr_bsr(self):

        for N in [1, 2, 3, 4, 5, 6, 10]:
            A = spdiags([2*ones(N), -ones(N), -ones(N)],
                        [0, -1, 1], N, N).tocsr()

            divisors = [n for n in range(1, N+1) if N % n == 0]

            x_csr = arange(N).astype(np.float64)
            b = x_csr**2
            gauss_seidel_nr(A, x_csr, b)

            for D in divisors:
                B = A.tobsr(blocksize=(D, D))
                x_bsr = arange(N).astype(np.float64)
                gauss_seidel_nr(B, x_bsr, b)
                assert_almost_equal(x_bsr, x_csr)

    def test_gauss_seidel_nr(self):
        scipy.random.seed(0)

        cases = []
        cases.append(poisson((4,), format='csr'))
        cases.append(poisson((4, 4), format='csr'))

        temp = asmatrix(rand(1, 1))
        cases.append(csr_matrix(temp))
        temp = asmatrix(rand(2, 2))
        cases.append(csr_matrix(temp))
        temp = asmatrix(rand(4, 4))
        cases.append(csr_matrix(temp))

        # reference implementation
        def gold(A, x, b, iterations, sweep):
            A = mat(A.todense())
            AA = A.H*A

            L = asmatrix(tril(AA, k=0))
            U = asmatrix(triu(AA, k=0))

            for i in range(iterations):
                if sweep == 'forward':
                    x = x + (solve(L, A.H*(b - A*x)))
                elif sweep == 'backward':
                    x = x + (solve(U, A.H*(b - A*x)))
                else:
                    x = x + (solve(L, A.H*(b - A*x)))
                    x = x + (solve(U, A.H*(b - A*x)))
            return x

        for A in cases:

            b = asmatrix(rand(A.shape[0], 1))
            x = asmatrix(rand(A.shape[0], 1))

            x_copy = x.copy()
            gauss_seidel_nr(A, x, b, iterations=1, sweep='forward')
            assert_almost_equal(x, gold(A, x_copy, b, iterations=1,
                                sweep='forward'))

            x_copy = x.copy()
            gauss_seidel_nr(A, x, b, iterations=1, sweep='backward')
            assert_almost_equal(x, gold(A, x_copy, b, iterations=1,
                                sweep='backward'))

            x_copy = x.copy()
            gauss_seidel_nr(A, x, b, iterations=1, sweep='symmetric')
            assert_almost_equal(x, gold(A, x_copy, b, iterations=1,
                                sweep='symmetric'))

        # forward and backward passes should give same result with
        # x=ones(N),b=zeros(N)
        N = 100
        A = spdiags([2*ones(N), -ones(N), -ones(N)], [0, -1, 1], N, N,
                    format='csr')
        x = ones(N)
        b = zeros(N)
        gauss_seidel_nr(A, x, b, iterations=200, sweep='forward')
        resid1 = np.linalg.norm(A*x, 2)
        x = ones(N)
        gauss_seidel_nr(A, x, b, iterations=200, sweep='backward')
        resid2 = np.linalg.norm(A*x, 2)
        self.assertTrue(resid1 < 0.2 and resid2 < 0.2)
        self.assertTrue(allclose(resid1, resid2))

    def test_schwarz_gold(self):
        scipy.random.seed(0)

        cases = []
        cases.append(poisson((4,), format='csr'))
        cases.append(poisson((4, 4), format='csr'))
        A = poisson((8, 8), format='csr')
        A.data[0] = 10.0
        A.data[1] = -0.5
        A.data[3] = -0.5
        cases.append(A)

        temp = asmatrix(rand(1, 1))
        cases.append(csr_matrix(temp.T * temp))
        temp = asmatrix(rand(2, 2))
        cases.append(csr_matrix(temp.T * temp))
        temp = asmatrix(rand(4, 4))
        cases.append(csr_matrix(temp.T * temp))

        # reference implementation
        def gold(A, x, b, iterations, sweep='forward'):
            A = csr_matrix(A)
            n = A.shape[0]

            # Default is point-wise iteration with each subdomain a point's
            # neighborhood in the matrix graph
            subdomains =\
                [A.indices[A.indptr[i]:A.indptr[i+1]] for i in range(n)]

            # extract each subdomain's block from the matrix
            subblocks = []
            for i in range(len(subdomains)):
                blkA = (A[subdomains[i], :]).tocsc()
                blkA = blkA[:, subdomains[i]].todense()
                blkAinv = scipy.linalg.pinv2(blkA)
                subblocks.append(blkAinv)

            if sweep == 'forward':
                indices = np.arange(len(subdomains))
            elif sweep == 'backward':
                indices = np.arange(len(subdomains)-1, -1, -1)
            elif sweep == 'symmetric':
                indices1 = np.arange(len(subdomains))
                indices2 = np.arange(len(subdomains)-1, -1, -1)
                indices = np.concatenate((indices1, indices2))

            # Multiplicative Schwarz iterations
            for j in range(iterations):
                for i in indices:
                    si = subdomains[i]
                    x[si] = scipy.dot(subblocks[i],
                                      (b[si] - A[si, :]*x)) + x[si]

            return x

        for A in cases:

            b = asmatrix(rand(A.shape[0], 1))
            x = asmatrix(rand(A.shape[0], 1))

            x_copy = x.copy()
            schwarz(A, x, b, iterations=1, sweep='forward')
            assert_almost_equal(x, gold(A, x_copy, b, iterations=1,
                                sweep='forward'))

            x_copy = x.copy()
            schwarz(A, x, b, iterations=1, sweep='backward')
            assert_almost_equal(x, gold(A, x_copy, b, iterations=1,
                                sweep='backward'))

            x_copy = x.copy()
            schwarz(A, x, b, iterations=1, sweep='symmetric')
            assert_almost_equal(x, gold(A, x_copy, b, iterations=1,
                                sweep='symmetric'))


# Test complex arithmetic
class TestComplexRelaxation(TestCase):
    def test_jacobi(self):
        N = 1
        A = spdiags([2*ones(N), -ones(N), -ones(N)], [0, -1, 1], N, N,
                    format='csr')
        A.data = A.data + 1.0j*A.data
        x = arange(N).astype(A.dtype)
        b = zeros(N).astype(A.dtype)
        jacobi(A, x, b)
        assert_almost_equal(x, array([0]))

        x = array([1.0 + 1.0j])
        b = array([-1.0 + 1.0j])
        omega = 4.0 - 1.0j
        jacobi(A, x, b, omega=omega)
        assert_almost_equal(x, array([-3.5]))

        N = 3
        A = spdiags([2*ones(N), -ones(N), -ones(N)], [0, -1, 1], N, N,
                    format='csr')
        A.data = A.data + 1.0j*A.data
        x = zeros(N).astype(A.dtype)
        b = arange(N).astype(A.dtype)
        b = b + 1.0j*b
        soln = array([0.0, 0.5, 1.0])
        jacobi(A, x, b)
        assert_almost_equal(x, soln)

        N = 3
        A = spdiags([2*ones(N), -ones(N), -ones(N)], [0, -1, 1], N, N,
                    format='csr')
        A.data = A.data + 1.0j*A.data
        x = arange(N).astype(A.dtype)
        x = x + 1.0j*x
        b = zeros(N).astype(A.dtype)
        soln = array([0.5 + 0.5j, 1.0 + 1.0j, 0.5+0.5j])
        jacobi(A, x, b)
        assert_almost_equal(x, soln)

        N = 1
        A = spdiags([2*ones(N), -ones(N), -ones(N)], [0, -1, 1], N, N,
                    format='csr')
        A.data = A.data + 1.0j*A.data
        x = arange(N).astype(A.dtype)
        b = array([10]).astype(A.dtype)
        jacobi(A, x, b)
        assert_almost_equal(x, array([2.5 - 2.5j]))

        N = 3
        A = spdiags([2*ones(N), -ones(N), -ones(N)], [0, -1, 1], N, N,
                    format='csr')
        A.data = A.data + 1.0j*A.data
        x = arange(N).astype(A.dtype)
        x = x + 1.0j*x
        b = array([10, 20, 30]).astype(A.dtype)
        soln = array([3.0-2.0j, 6.0-4.0j, 8.0-7.0j])
        jacobi(A, x, b)
        assert_almost_equal(x, soln)

        N = 3
        A = spdiags([2*ones(N), -ones(N), -ones(N)], [0, -1, 1], N, N,
                    format='csr')
        A.data = A.data + 1.0j*A.data
        x = arange(N).astype(A.dtype)
        x = x + 1.0j*x
        x_copy = x.copy()
        b = array([10, 20, 30]).astype(A.dtype)
        soln = 2.0/3.0*x_copy + 1.0/3.0*array([3.0-2.0j, 6.0-4.0j, 8.0-7.0j])
        jacobi(A, x, b, omega=1.0/3.0)
        assert_almost_equal(x, soln)

    def test_jacobi_bsr(self):
        cases = []
        for N in [1, 2, 3, 4, 5, 6, 10]:
            #
            C = spdiags([2*ones(N), -ones(N), -ones(N)], [0, -1, 1],
                        N, N).tocsr()
            C.data = C.data + 1.0j*1e-3*rand(C.data.shape[0],)
            cases.append(C)
            #
            cases.append(1.0j*elasticity.linear_elasticity((N, N))[0].tocsr())
            #
            C = csr_matrix(rand(N, N) + 1.0j*rand(N, N))
            cases.append(C*C.H)
            #
            C = sprand(N*2, N*2, 0.3) + 1.0j*sprand(N*2, N*2, 0.3) +\
                eye(N*2, N*2)
            cases.append(C*C.H)

        for A in cases:
            n = A.shape[0]
            divisors = [i for i in range(1, n+1) if n % i == 0]

            x0 = (arange(n) + 1.0j*1e-3*rand(n,)).astype(A.dtype)
            x_csr = x0.copy()
            b = x_csr**2
            jacobi(A, x_csr, b)

            for D in divisors:
                B = A.tobsr(blocksize=(D, D))
                x_bsr = x0.copy()
                jacobi(B, x_bsr, b)
                assert_almost_equal(x_bsr, x_csr)

    def test_gauss_seidel_bsr(self):
        sweeps = ['forward', 'backward', 'symmetric']
        cases = []
        for N in [1, 2, 3, 4, 5, 6, 10]:
            #
            C = spdiags([2*ones(N), -ones(N), -ones(N)], [0, -1, 1],
                        N, N).tocsr()
            C.data = C.data + 1.0j*1e-3*rand(C.data.shape[0],)
            cases.append(C)
            #
            cases.append(1.0j*elasticity.linear_elasticity((N, N))[0].tocsr())
            #
            C = csr_matrix(rand(N, N) + 1.0j*rand(N, N))
            cases.append(C*C.H)
            #
            C = sprand(N*2, N*2, 0.3) + 1.0j*sprand(N*2, N*2, 0.3) +\
                eye(N*2, N*2)
            cases.append(C*C.H)

        for A in cases:
            n = A.shape[0]
            for sweep in sweeps:
                divisors = [i for i in range(1, n+1) if n % i == 0]

                x0 = (arange(n) + 1.0j*1e-3*rand(n,)).astype(A.dtype)
                x_csr = x0.copy()
                b = x_csr**2
                gauss_seidel(A, x_csr, b, sweep=sweep)

                for D in divisors:
                    B = A.tobsr(blocksize=(D, D))
                    x_bsr = x0.copy()
                    gauss_seidel(B, x_bsr, b, sweep=sweep)
                    assert_almost_equal(x_bsr, x_csr)

    def test_schwarz_gold(self):
        scipy.random.seed(0)

        cases = []
        A = poisson((4,), format='csr')
        A.data = A.data + 1.0j*1e-3*rand(A.data.shape[0],)
        cases.append(A)
        A = poisson((4, 4), format='csr')
        A.data = A.data + 1.0j*1e-3*rand(A.data.shape[0],)
        cases.append(A)

        temp = asmatrix(rand(1, 1)) + 1.0j*asmatrix(rand(1, 1))
        cases.append(csr_matrix(temp.H * temp))
        temp = asmatrix(rand(2, 2)) + 1.0j*asmatrix(rand(2, 2))
        cases.append(csr_matrix(temp.H * temp))
        temp = asmatrix(rand(4, 4)) + 1.0j*asmatrix(rand(4, 4))
        cases.append(csr_matrix(temp.H * temp))

        # reference implementation
        def gold(A, x, b, iterations):
            A = csr_matrix(A)
            n = A.shape[0]

            # Default is point-wise iteration with each subdomain a point's
            # neighborhood in the matrix graph
            subdomains =\
                [A.indices[A.indptr[i]:A.indptr[i+1]] for i in range(n)]

            # extract each subdomain's block from the matrix
            subblocks = []
            for i in range(len(subdomains)):
                blkA = (A[subdomains[i], :]).tocsc()
                blkA = blkA[:, subdomains[i]].todense()
                blkAinv = scipy.linalg.pinv2(blkA)
                subblocks.append(blkAinv)

            # Multiplicative Schwarz iterations
            for j in range(iterations):
                for i in range(len(subdomains)):
                    si = subdomains[i]
                    x[si] = scipy.dot(subblocks[i],
                                      (b[si] - A[si, :]*x)) + x[si]
            return x

        for A in cases:

            n = A.shape[0]
            b = asmatrix(rand(n, 1)) +\
                1.0j*asmatrix(rand(n, 1)).astype(A.dtype)
            x = asmatrix(rand(n, 1)) +\
                1.0j*asmatrix(rand(n, 1)).astype(A.dtype)

            x_copy = x.copy()
            schwarz(A, x, b, iterations=1)
            assert_almost_equal(x, gold(A, x_copy, b, iterations=1))

    def test_gauss_seidel_new(self):
        scipy.random.seed(0)

        cases = []
        A = poisson((4,), format='csr')
        A.data = A.data + 1.0j*1e-3*rand(A.data.shape[0],)
        cases.append(A)
        A = poisson((4, 4), format='csr')
        A.data = A.data + 1.0j*1e-3*rand(A.data.shape[0],)
        cases.append(A)

        temp = asmatrix(rand(4, 4)) + 1.0j*asmatrix(rand(4, 4))
        cases.append(csr_matrix(temp.H * temp))

        # reference implementation
        def gold(A, x, b, iterations, sweep):
            A = A.todense()

            L = asmatrix(tril(A, k=-1))
            D = asmatrix(diag(diag(A)))
            U = asmatrix(triu(A, k=1))

            for i in range(iterations):
                if sweep == 'forward':
                    x = solve(L + D, (b - U*x))
                elif sweep == 'backward':
                    x = solve(U + D, (b - L*x))
                else:
                    x = solve(L + D, (b - U*x))
                    x = solve(U + D, (b - L*x))
            return x

        for A in cases:
            n = A.shape[0]
            b = asmatrix(rand(n, 1)) +\
                1.0j*asmatrix(rand(n, 1)).astype(A.dtype)
            x = asmatrix(rand(n, 1)) +\
                1.0j*asmatrix(rand(n, 1)).astype(A.dtype)

            # Gauss-Seidel Tests
            x_copy = x.copy()
            gauss_seidel(A, x, b, iterations=1, sweep='forward')
            assert_almost_equal(x, gold(A, x_copy, b, iterations=1,
                                sweep='forward'))

            x_copy = x.copy()
            gauss_seidel(A, x, b, iterations=1, sweep='backward')
            assert_almost_equal(x, gold(A, x_copy, b, iterations=1,
                                sweep='backward'))

            x_copy = x.copy()
            gauss_seidel(A, x, b, iterations=1, sweep='symmetric')
            assert_almost_equal(x, gold(A, x_copy, b, iterations=1,
                                sweep='symmetric'))

            ##
            # Indexed Gauss-Seidel Tests
            x_copy = x.copy()
            gauss_seidel_indexed(A, x, b, indices=arange(A.shape[0]),
                                 iterations=1, sweep='forward')
            assert_almost_equal(x, gold(A, x_copy, b, iterations=1,
                                sweep='forward'))

            x_copy = x.copy()
            gauss_seidel_indexed(A, x, b, indices=arange(A.shape[0]),
                                 iterations=1, sweep='backward')
            assert_almost_equal(x, gold(A, x_copy, b, iterations=1,
                                sweep='backward'))

            x_copy = x.copy()
            gauss_seidel_indexed(A, x, b, indices=arange(A.shape[0]),
                                 iterations=1, sweep='symmetric')
            assert_almost_equal(x, gold(A, x_copy, b, iterations=1,
                                sweep='symmetric'))

    def test_gauss_seidel_csr(self):
        N = 1
        A = spdiags([2*ones(N), -ones(N), -ones(N)], [0, -1, 1], N, N,
                    format='csr')
        A.data = A.data + 1.0j*A.data
        x = arange(N).astype(A.dtype)
        x = x + 1.0j*x
        b = zeros(N).astype(A.dtype)
        gauss_seidel(A, x, b)
        assert_almost_equal(x, array([0]))

        N = 3
        A = spdiags([2*ones(N), -ones(N), -ones(N)], [0, -1, 1], N, N,
                    format='csr')
        A.data = A.data + 1.0j*A.data
        x = arange(N).astype(A.dtype)
        x = x + 1.0j*x
        soln = array([1.0/2.0, 5.0/4.0, 5.0/8.0])
        soln = soln + 1.0j*soln
        b = zeros(N).astype(A.dtype)
        gauss_seidel(A, x, b)
        assert_almost_equal(x, soln)

        N = 1
        A = spdiags([2*ones(N), -ones(N), -ones(N)], [0, -1, 1], N, N,
                    format='csr')
        A.data = A.data + 1.0j*A.data
        x = arange(N).astype(A.dtype)
        x = x + 1.0j*x
        b = zeros(N).astype(A.dtype)
        gauss_seidel(A, x, b, sweep='backward')
        assert_almost_equal(x, array([0]))

        N = 3
        A = spdiags([2*ones(N), -ones(N), -ones(N)], [0, -1, 1], N, N,
                    format='csr')
        A.data = A.data + 1.0j*A.data
        x = arange(N).astype(A.dtype)
        x = x + 1.0j*x
        soln = array([1.0/8.0, 1.0/4.0, 1.0/2.0])
        soln = soln + 1.0j*soln
        b = zeros(N).astype(A.dtype)
        gauss_seidel(A, x, b, sweep='backward')
        assert_almost_equal(x, soln)

        N = 1
        A = spdiags([2*ones(N), -ones(N), -ones(N)], [0, -1, 1], N, N,
                    format='csr')
        A.data = A.data + 1.0j*A.data
        x = arange(N).astype(A.dtype)
        b = array([10]).astype(A.dtype)
        gauss_seidel(A, x, b)
        assert_almost_equal(x, array([2.5 - 2.5j]))

        N = 3
        A = spdiags([2*ones(N), -ones(N), -ones(N)], [0, -1, 1], N, N,
                    format='csr')
        A.data = A.data + 1.0j*A.data
        soln = array([3.0 - 2.0j, 7.5 - 5.0j, 11.25 - 10.0j])
        x = arange(N).astype(A.dtype)
        x = x + 1.0j*x
        b = array([10, 20, 30]).astype(A.dtype)
        gauss_seidel(A, x, b)
        assert_almost_equal(x, soln)

        # forward and backward passes should give same result with
        # x=ones(N),b=zeros(N)
        N = 100
        A = spdiags([2*ones(N), -ones(N), -ones(N)], [0, -1, 1], N, N,
                    format='csr')
        A.data = A.data + 1.0j*A.data
        x = ones(N).astype(A.dtype)
        x = x + 1.0j*x
        b = zeros(N).astype(A.dtype)
        gauss_seidel(A, x, b, iterations=200, sweep='forward')
        resid1 = np.linalg.norm(A*x, 2)
        x = ones(N).astype(A.dtype)
        x = x + 1.0j*x
        gauss_seidel(A, x, b, iterations=200, sweep='backward')
        resid2 = np.linalg.norm(A*x, 2)
        self.assertTrue(resid1 < 0.03 and resid2 < 0.03)
        self.assertTrue(allclose(resid1, resid2))

    def test_jacobi_ne(self):
        N = 1
        A = spdiags([2*ones(N), -ones(N), -ones(N)], [0, -1, 1], N, N,
                    format='csr')
        A.data = A.data + 1.0j*A.data
        x = arange(N).astype(A.dtype)
        b = zeros(N).astype(A.dtype)
        jacobi_ne(A, x, b)
        assert_almost_equal(x, array([0]))

        N = 3
        A = spdiags([2*ones(N), -ones(N), -ones(N)], [0, -1, 1], N, N,
                    format='csr')
        A.data = A.data + 1.0j*A.data
        soln = array([-1./6., -1./15., 19./30.])
        x = zeros(N).astype(A.dtype)
        b = arange(N).astype(A.dtype)
        b = b + 1.0j*b
        jacobi_ne(A, x, b)
        assert_almost_equal(x, soln)

        N = 3
        A = spdiags([2*ones(N), -ones(N), -ones(N)], [0, -1, 1], N, N,
                    format='csr')
        A.data = A.data + 1.0j*A.data
        soln = array([2./5. + 2.0j/5., 7./5. + 7.0j/5., 4./5. + 4.0j/5.])
        x = arange(N).astype(A.dtype)
        x = x + 1.0j*x
        b = zeros(N).astype(A.dtype)
        jacobi_ne(A, x, b)
        assert_almost_equal(x, soln)

        N = 1
        A = spdiags([2*ones(N), -ones(N), -ones(N)], [0, -1, 1], N, N,
                    format='csr')
        A.data = A.data + 1.0j*A.data
        x = arange(N).astype(A.dtype)
        b = array([10]).astype(A.dtype)
        jacobi_ne(A, x, b)
        assert_almost_equal(x, array([2.5 - 2.5j]))

        N = 3
        A = spdiags([2*ones(N), -ones(N), -ones(N)], [0, -1, 1], N, N,
                    format='csr')
        A.data = A.data + 1.0j*A.data
        soln = array([11./15. + 1.0j/15., 11./15. +
                      31.0j/15, 77./15. - 53.0j/15.])
        x = arange(N).astype(A.dtype)
        x = x + 1.0j*x
        b = array([10, 20, 30]).astype(A.dtype)
        jacobi_ne(A, x, b)
        assert_almost_equal(x, soln)

        N = 3
        A = spdiags([2*ones(N), -ones(N), -ones(N)], [0, -1, 1], N, N,
                    format='csr')
        A.data = A.data + 1.0j*A.data
        x = arange(N).astype(A.dtype)
        x = x + 1.0j*x
        b = array([10, 20, 30]).astype(A.dtype)
        x_copy = x.copy()
        solnpart = array([11./15. + 1.0j/15., 11./15. +
                          31.0j/15, 77./15. - 53.0j/15.])
        soln = 2.0/3.0*x_copy + 1.0/3.0*solnpart

        jacobi_ne(A, x, b, omega=1.0/3.0)
        assert_almost_equal(x, soln)

    def test_gauss_seidel_ne_bsr(self):
        for N in [1, 2, 3, 4, 5, 6, 10]:
            A = spdiags([2*ones(N), -ones(N), -ones(N)], [0, -1, 1],
                        N, N).tocsr()
            A.data = A.data + 1.0j*1e-3*rand(A.data.shape[0],)

            divisors = [n for n in range(1, N+1) if N % n == 0]

            x_csr = (arange(N) + 1.0j*1e-3*rand(N,)).astype(A.dtype)
            x_bsr = x_csr.copy()
            b = x_csr**2
            gauss_seidel_ne(A, x_csr, b)

            for D in divisors:
                B = A.tobsr(blocksize=(D, D))
                x_bsr_temp = x_bsr.copy()
                gauss_seidel_ne(B, x_bsr_temp, b)
                assert_almost_equal(x_bsr_temp, x_csr)

    def test_gauss_seidel_ne_new(self):
        scipy.random.seed(0)

        cases = []
        A = poisson((4,), format='csr')
        A.data = A.data + 1.0j*1e-3*rand(A.data.shape[0],)
        cases.append(A)
        A = poisson((4, 4), format='csr')
        A.data = A.data + 1.0j*1e-3*rand(A.data.shape[0],)
        cases.append(A.tobsr(blocksize=(2, 2)))

        temp = asmatrix(rand(4, 4)) + 1.0j*asmatrix(rand(4, 4))
        cases.append(csr_matrix(temp.T * temp))

        # reference implementation
        def gold(A, x, b, iterations, sweep):
            A = mat(A.todense())
            AA = A*A.H

            L = asmatrix(tril(AA, k=0))
            U = asmatrix(triu(AA, k=0))

            for i in range(iterations):
                if sweep == 'forward':
                    x = x + A.H*(solve(L, (b - A*x)))
                elif sweep == 'backward':
                    x = x + A.H*(solve(U, (b - A*x)))
                else:
                    x = x + A.H*(solve(L, (b - A*x)))
                    x = x + A.H*(solve(U, (b - A*x)))
            return x

        for A in cases:
            n = A.shape[0]
            b = asmatrix(rand(n, 1)) +\
                1.0j*asmatrix(rand(n, 1)).astype(A.dtype)
            x = asmatrix(rand(n, 1)) +\
                1.0j*asmatrix(rand(n, 1)).astype(A.dtype)

            x_copy = x.copy()
            gauss_seidel_ne(A, x, b, iterations=1, sweep='forward')
            assert_almost_equal(x, gold(A, x_copy, b, iterations=1,
                                sweep='forward'))

            x_copy = x.copy()
            gauss_seidel_ne(A, x, b, iterations=1, sweep='backward')
            assert_almost_equal(x, gold(A, x_copy, b, iterations=1,
                                sweep='backward'))

            x_copy = x.copy()
            gauss_seidel_ne(A, x, b, iterations=1, sweep='symmetric')
            assert_almost_equal(x, gold(A, x_copy, b, iterations=1,
                                sweep='symmetric'))

    def test_gauss_seidel_ne_csr(self):
        N = 1
        A = spdiags([2*ones(N), -ones(N), -ones(N)], [0, -1, 1],
                    N, N, format='csr')
        x = arange(N).astype(A.dtype)
        x = x + 1.0j*x
        A.data = A.data + 1.0j*A.data
        b = zeros(N).astype(A.dtype)
        gauss_seidel_ne(A, x, b)
        assert_almost_equal(x, array([0]))

        N = 3
        A = spdiags([2*ones(N), -ones(N), -ones(N)], [0, -1, 1],
                    N, N, format='csr')
        A.data = A.data + 1.0j*A.data
        soln = array([4./15., 8./5., 4./5.])
        soln = soln + 1.0j*soln
        x = arange(N).astype(A.dtype)
        x = x + 1.0j*x
        b = zeros(N).astype(A.dtype)
        gauss_seidel_ne(A, x, b)
        assert_almost_equal(x, soln)

        N = 1
        A = spdiags([2*ones(N), -ones(N), -ones(N)], [0, -1, 1],
                    N, N, format='csr')
        A.data = A.data + 1.0j*A.data
        x = arange(N).astype(A.dtype)
        x = x + 1.0j*x
        b = zeros(N).astype(A.dtype)
        gauss_seidel_ne(A, x, b, sweep='backward')
        assert_almost_equal(x, array([0]))

        N = 3
        A = spdiags([2*ones(N), -ones(N), -ones(N)], [0, -1, 1],
                    N, N, format='csr')
        A.data = A.data + 1.0j*A.data
        soln = array([2./5., 4./5., 6./5.])
        soln = soln + 1.0j*soln
        x = arange(N).astype(A.dtype)
        x = x + 1.0j*x
        b = zeros(N).astype(A.dtype)
        gauss_seidel_ne(A, x, b, sweep='backward')
        assert_almost_equal(x, soln)

        N = 1
        A = spdiags([2*ones(N), -ones(N), -ones(N)], [0, -1, 1],
                    N, N, format='csr')
        A.data = A.data + 1.0j*A.data
        x = arange(N).astype(A.dtype)
        b = array([10]).astype(A.dtype)
        gauss_seidel_ne(A, x, b)
        assert_almost_equal(x, array([2.5 - 2.5j]))

        N = 3
        A = spdiags([2*ones(N), -ones(N), -ones(N)], [0, -1, 1],
                    N, N, format='csr')
        A.data = A.data + 1.0j*A.data
        soln = array([-1./15.+0.6j, 0.6+2.6j, 7.8-6.2j])
        x = arange(N).astype(A.dtype)
        x = x + 1.0j*x
        b = array([10, 20, 30]).astype(A.dtype)
        gauss_seidel_ne(A, x, b)
        assert_almost_equal(x, soln)

        # forward and backward passes should give same result with
        # x=ones(N),b=zeros(N)
        N = 100
        A = spdiags([2*ones(N), -ones(N), -ones(N)], [0, -1, 1],
                    N, N, format='csr')
        A.data = A.data + 1.0j*A.data
        x = ones(N).astype(A.dtype)
        x = x + 1.0j*x
        b = zeros(N).astype(A.dtype)
        gauss_seidel_ne(A, x, b, iterations=200, sweep='forward')
        resid1 = np.linalg.norm(A*x, 2)

        x = ones(N).astype(A.dtype)
        x = x + 1.0j*x
        gauss_seidel_ne(A, x, b, iterations=200, sweep='backward')
        resid2 = np.linalg.norm(A*x, 2)
        self.assertTrue(resid1 < 0.3 and resid2 < 0.3)
        self.assertTrue(allclose(resid1, resid2))

    def test_gauss_seidel_nr_bsr(self):
        for N in [1, 2, 3, 4, 5, 6, 10]:
            A = spdiags([2*ones(N), -ones(N), -ones(N)], [0, -1, 1],
                        N, N).tocsr()
            A.data = A.data + 1.0j*1e-3*rand(A.data.shape[0],)

            divisors = [n for n in range(1, N+1) if N % n == 0]

            x_csr = (arange(N) + 1.0j*1e-3*rand(N,)).astype(A.dtype)
            x_bsr = x_csr.copy()
            b = x_csr**2
            gauss_seidel_nr(A, x_csr, b)

            for D in divisors:
                B = A.tobsr(blocksize=(D, D))
                x_bsr_temp = x_bsr.copy()
                gauss_seidel_nr(B, x_bsr_temp, b)
                assert_almost_equal(x_bsr_temp, x_csr)

    def test_gauss_seidel_nr(self):
        scipy.random.seed(0)

        cases = []
        A = poisson((4,), format='csr')
        A.data = A.data + 1.0j*1e-3*rand(A.data.shape[0],)
        cases.append(A)
        A = poisson((4, 4), format='csr')
        A.data = A.data + 1.0j*1e-3*rand(A.data.shape[0],)
        cases.append(A.tobsr(blocksize=(2, 2)))

        temp = asmatrix(rand(1, 1)) + 1.0j*asmatrix(rand(1, 1))
        cases.append(csr_matrix(temp))
        temp = asmatrix(rand(2, 2)) + 1.0j*asmatrix(rand(2, 2))
        cases.append(csr_matrix(temp))
        temp = asmatrix(rand(4, 4)) + 1.0j*asmatrix(rand(4, 4))
        cases.append(csr_matrix(temp))

        # reference implementation
        def gold(A, x, b, iterations, sweep):
            A = mat(A.todense())
            AA = A.H*A

            L = asmatrix(tril(AA, k=0))
            U = asmatrix(triu(AA, k=0))

            for i in range(iterations):
                if sweep == 'forward':
                    x = x + (solve(L, A.H*(b - A*x)))
                elif sweep == 'backward':
                    x = x + (solve(U, A.H*(b - A*x)))
                else:
                    x = x + (solve(L, A.H*(b - A*x)))
                    x = x + (solve(U, A.H*(b - A*x)))
            return x

        for A in cases:
            n = A.shape[0]
            b = asmatrix(rand(n, 1)) +\
                1.0j*asmatrix(rand(n, 1)).astype(A.dtype)
            x = asmatrix(rand(n, 1)) +\
                1.0j*asmatrix(rand(n, 1)).astype(A.dtype)

            x_copy = x.copy()
            gauss_seidel_nr(A, x, b, iterations=1, sweep='forward')
            assert_almost_equal(x, gold(A, x_copy, b, iterations=1,
                                sweep='forward'))

            x_copy = x.copy()
            gauss_seidel_nr(A, x, b, iterations=1, sweep='backward')
            assert_almost_equal(x, gold(A, x_copy, b, iterations=1,
                                sweep='backward'))

            x_copy = x.copy()
            gauss_seidel_nr(A, x, b, iterations=1, sweep='symmetric')
            assert_almost_equal(x, gold(A, x_copy, b, iterations=1,
                                sweep='symmetric'))

        # forward and backward passes should give same result with
        # x=ones(N),b=zeros(N)
        N = 100
        A = spdiags([2*ones(N), -ones(N), -ones(N)], [0, -1, 1],
                    N, N, format='csr')
        A.data = A.data + 1.0j*A.data
        x = ones(N).astype(A.dtype)
        x = x + 1.0j*x
        b = zeros(N).astype(A.dtype)
        gauss_seidel_nr(A, x, b, iterations=200, sweep='forward')
        resid1 = np.linalg.norm(A*x, 2)

        x = ones(N).astype(A.dtype)
        x = x + 1.0j*x
        gauss_seidel_nr(A, x, b, iterations=200, sweep='backward')
        resid2 = np.linalg.norm(A*x, 2)
        self.assertTrue(resid1 < 0.3 and resid2 < 0.3)
        self.assertTrue(allclose(resid1, resid2))


# Test both complex and real arithmetic
# for block_jacobi and block_gauss_seidel
class TestBlockRelaxation(TestCase):

    def test_block_jacobi(self):
        scipy.random.seed(0)

        # All real valued tests
        cases = []
        A = csr_matrix(scipy.zeros((1, 1)))
        cases.append((A, 1))
        A = csr_matrix(scipy.rand(1, 1))
        cases.append((A, 1))
        A = csr_matrix(scipy.zeros((2, 2)))
        cases.append((A, 1))
        cases.append((A, 2))
        A = csr_matrix(scipy.rand(2, 2))
        cases.append((A, 1))
        cases.append((A, 2))
        A = csr_matrix(scipy.zeros((3, 3)))
        cases.append((A, 1))
        cases.append((A, 3))
        A = csr_matrix(scipy.rand(3, 3))
        cases.append((A, 1))
        cases.append((A, 3))
        A = poisson((4, 4), format='csr')
        cases.append((A, 1))
        cases.append((A, 2))
        cases.append((A, 4))
        A = array([[9.1, 9.8, 9.6, 0., 3.6, 0.],
                   [18.2, 19.6, 0., 0., 1.7, 2.8],
                   [0., 0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0., 0.],
                   [0., 0., 0., 4.2, 1., 1.1],
                   [0., 0., 9.1, 0., 0., 9.3]])
        A = csr_matrix(A)
        cases.append((A, 1))
        cases.append((A, 2))
        cases.append((A, 3))

        # reference implementation of 1 iteration
        def gold(A, x, b, blocksize, omega):

            A = csr_matrix(A)
            temp = x.copy()
            Dinv = get_block_diag(A, blocksize=blocksize, inv_flag=True)
            D = get_block_diag(A, blocksize=blocksize, inv_flag=False)
            I0 = scipy.arange(Dinv.shape[0])
            I1 = scipy.arange(Dinv.shape[0] + 1)
            A_no_D = A - bsr_matrix((D, I0, I1), shape=A.shape)
            A_no_D = csr_matrix(A_no_D)

            for i in range(0, A.shape[0], blocksize):
                r = A_no_D[i:(i+blocksize), :]*temp
                r = scipy.mat(Dinv[int(i/blocksize), :, :]) *\
                    scipy.mat(scipy.ravel(b[i:(i+blocksize)]) -
                              scipy.ravel(r)).reshape(-1, 1)
                x[i:(i+blocksize)] = (1.0 - omega)*temp[i:(i+blocksize)] +\
                    omega*scipy.ravel(r)

            return x

        for A, blocksize in cases:
            b = rand(A.shape[0])
            x = rand(A.shape[0])
            x_copy = x.copy()
            block_jacobi(A, x, b, blocksize=blocksize, iterations=1, omega=1.1)
            assert_almost_equal(x, gold(A, x_copy, b, blocksize, 1.1),
                                decimal=4)

        # check for agreement between jacobi and block jacobi with blocksize=1
        A = poisson((4, 5), format='csr')
        b = rand(A.shape[0])
        x = rand(A.shape[0])
        x_copy = x.copy()
        block_jacobi(A, x, b, blocksize=1, iterations=2, omega=1.1)
        jacobi(A, x_copy, b, iterations=2, omega=1.1)
        assert_almost_equal(x, x_copy, decimal=4)

        # complex valued tests
        cases = []
        A = csr_matrix(scipy.rand(3, 3) + 1.0j*scipy.rand(3, 3))
        cases.append((A, 1))
        cases.append((A, 3))
        A = poisson((4, 4), format='csr')
        A.data = A.data + 1.0j*scipy.rand(A.data.shape[0])
        cases.append((A, 1))
        cases.append((A, 2))
        cases.append((A, 4))
        A = array([[9.1j, 9.8j, 9.6, 0., 3.6, 0.],
                   [18.2j, 19.6j, 0., 0., 1.7, 2.8],
                   [0., 0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0., 0.],
                   [0., 0., 0., 4.2, 1.0j, 1.1],
                   [0., 0., 9.1, 0., 0., 9.3]])
        A = csr_matrix(A)
        cases.append((A, 1))
        cases.append((A, 2))
        cases.append((A, 3))

        for A, blocksize in cases:
            b = rand(A.shape[0]) + 1.0j*rand(A.shape[0])
            x = rand(A.shape[0]) + 1.0j*rand(A.shape[0])
            x_copy = x.copy()
            block_jacobi(A, x, b, blocksize=blocksize, iterations=1, omega=1.1)
            assert_almost_equal(x, gold(A, x_copy, b, blocksize, 1.1),
                                decimal=4)

    def test_block_gauss_seidel(self):
        scipy.random.seed(0)

        # All real valued tests
        cases = []
        A = csr_matrix(scipy.zeros((1, 1)))
        cases.append((A, 1))
        A = csr_matrix(scipy.rand(1, 1))
        cases.append((A, 1))
        A = csr_matrix(scipy.zeros((2, 2)))
        cases.append((A, 1))
        cases.append((A, 2))
        A = csr_matrix(scipy.rand(2, 2))
        cases.append((A, 1))
        cases.append((A, 2))
        A = csr_matrix(scipy.zeros((3, 3)))
        cases.append((A, 1))
        cases.append((A, 3))
        A = csr_matrix(scipy.rand(3, 3))
        cases.append((A, 1))
        cases.append((A, 3))
        A = poisson((4, 4), format='csr')
        cases.append((A, 1))
        cases.append((A, 2))
        cases.append((A, 4))
        A = array([[9.1, 9.8, 9.6, 0., 3.6, 0.],
                   [18.2, 19.6, 0., 0., 1.7, 2.8],
                   [0., 0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0., 0.],
                   [0., 0., 0., 4.2, 1., 1.1],
                   [0., 0., 9.1, 0., 0., 9.3]])
        A = csr_matrix(A)
        cases.append((A, 1))
        cases.append((A, 2))
        cases.append((A, 3))

        # reference implementation of 1 iteration
        def gold(A, x, b, blocksize, sweep):

            A = csr_matrix(A)
            Dinv = get_block_diag(A, blocksize=blocksize, inv_flag=True)
            D = get_block_diag(A, blocksize=blocksize, inv_flag=False)
            I0 = scipy.arange(Dinv.shape[0])
            I1 = scipy.arange(Dinv.shape[0] + 1)
            A_no_D = A - bsr_matrix((D, I0, I1), shape=A.shape)
            A_no_D = csr_matrix(A_no_D)

            if sweep == 'symmetric':
                x = gold(A, x, b, blocksize, 'forward')
                x = gold(A, x, b, blocksize, 'backward')
                return x
            elif sweep == 'forward':
                start, stop, step = (0, A.shape[0], blocksize)
            elif sweep == 'backward':
                start, stop, step =\
                    (A.shape[0] - blocksize, -blocksize, -blocksize)

            for i in range(start, stop, step):
                r = A_no_D[i:(i+blocksize), :]*x
                r = scipy.mat(Dinv[int(i/blocksize), :, :]) *\
                    scipy.mat(scipy.ravel(b[i:(i+blocksize)]) -
                              scipy.ravel(r)).reshape(-1, 1)
                x[i:(i+blocksize)] = scipy.ravel(r)

            return x

        for A, blocksize in cases:
            b = rand(A.shape[0])
            # forward
            x = rand(A.shape[0])
            x_copy = x.copy()
            block_gauss_seidel(A, x, b, iterations=1, sweep='forward',
                               blocksize=blocksize)
            assert_almost_equal(x, gold(A, x_copy, b, blocksize, 'forward'),
                                decimal=4)
            # backward
            x = rand(A.shape[0])
            x_copy = x.copy()
            block_gauss_seidel(A, x, b, iterations=1, sweep='backward',
                               blocksize=blocksize)
            assert_almost_equal(x, gold(A, x_copy, b, blocksize, 'backward'),
                                decimal=4)
            # symmetric
            x = rand(A.shape[0])
            x_copy = x.copy()
            block_gauss_seidel(A, x, b, iterations=1, sweep='symmetric',
                               blocksize=blocksize)
            assert_almost_equal(x, gold(A, x_copy, b, blocksize, 'symmetric'),
                                decimal=4)

        # check for aggreement between gauss_seidel and block gauss-seidel
        # with blocksize=1
        A = poisson((4, 5), format='csr')
        b = rand(A.shape[0])
        # forward
        x = rand(A.shape[0])
        x_copy = x.copy()
        block_gauss_seidel(A, x, b, iterations=2, sweep='forward',
                           blocksize=1)
        gauss_seidel(A, x_copy, b, iterations=2, sweep='forward')
        assert_almost_equal(x, x_copy, decimal=4)
        # backward
        x = rand(A.shape[0])
        x_copy = x.copy()
        block_gauss_seidel(A, x, b, iterations=2, sweep='backward',
                           blocksize=1)
        gauss_seidel(A, x_copy, b, iterations=2, sweep='backward')
        assert_almost_equal(x, x_copy, decimal=4)
        # symmetric
        x = rand(A.shape[0])
        x_copy = x.copy()
        block_gauss_seidel(A, x, b, iterations=2, sweep='symmetric',
                           blocksize=1)
        gauss_seidel(A, x_copy, b, iterations=2, sweep='symmetric')
        assert_almost_equal(x, x_copy, decimal=4)

        # complex valued tests
        cases = []
        A = csr_matrix(scipy.rand(3, 3) + 1.0j*scipy.rand(3, 3))
        cases.append((A, 1))
        cases.append((A, 3))
        A = poisson((4, 4), format='csr')
        A.data = A.data + 1.0j*scipy.rand(A.data.shape[0])
        cases.append((A, 1))
        cases.append((A, 2))
        cases.append((A, 4))
        A = array([[9.1j, 9.8j, 9.6, 0., 3.6, 0.],
                   [18.2j, 19.6j, 0., 0., 1.7, 2.8],
                   [0., 0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0., 0.],
                   [0., 0., 0., 4.2, 1.0j, 1.1],
                   [0., 0., 9.1, 0., 0., 9.3]])
        A = csr_matrix(A)
        cases.append((A, 1))
        cases.append((A, 2))
        cases.append((A, 3))

        for A, blocksize in cases:
            b = rand(A.shape[0]) + 1.0j*rand(A.shape[0])
            # forward
            x = rand(A.shape[0]) + 1.0j*rand(A.shape[0])
            x_copy = x.copy()
            block_gauss_seidel(A, x, b, iterations=1, sweep='forward',
                               blocksize=blocksize)
            assert_almost_equal(x, gold(A, x_copy, b, blocksize, 'forward'),
                                decimal=4)
            # backward
            x = rand(A.shape[0]) + 1.0j*rand(A.shape[0])
            x_copy = x.copy()
            block_gauss_seidel(A, x, b, iterations=1, sweep='backward',
                               blocksize=blocksize)
            assert_almost_equal(x, gold(A, x_copy, b, blocksize, 'backward'),
                                decimal=4)
            # symmetric
            x = rand(A.shape[0]) + 1.0j*rand(A.shape[0])
            x_copy = x.copy()
            block_gauss_seidel(A, x, b, iterations=1, sweep='symmetric',
                               blocksize=blocksize)
            assert_almost_equal(x, gold(A, x_copy, b, blocksize, 'symmetric'),
                                decimal=4)

# class TestDispatch(TestCase):
#     def test_string(self):
#         from pyamg.relaxation import dispatch
#
#         A = poisson( (4,), format='csr')
#
#         cases = []
#         cases.append( 'gauss_seidel' )
#         cases.append( ('gauss_seidel',{'iterations':3}) )
#
#         for case in cases:
#             fn = dispatch(case)
#             fn(A, ones(4), zeros(4))
